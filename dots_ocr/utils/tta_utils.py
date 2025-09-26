# dots_ocr/utils/tta_utils.py
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

from PIL import Image
import torch

from qwen_vl_utils import process_vision_info
from dots_ocr.utils.image_utils import fetch_image
from dots_ocr.utils.output_cleaner import OutputCleaner
from dots_ocr.utils.consts import MIN_PIXELS
TTA_LANDSCAPE_LIMIT = 5
_tta_landscape_count = 0
@dataclass
class Cell:
    category: str
    bbox: List[int]          # [x1,y1,x2,y2] in ORIG image coords
    order_hi: int | None     # model output order at hi-scale (list index)
    order_lo: int | None     # model output order at lo-scale (list index)
    source: str              # 'hi' | 'lo' | 'both'
    confidence_score: float  # for mAP sorting only

def is_landscape(img: Image.Image, thr: float = 1.0) -> bool:
    """w/h >= thr이면 가로형으로 간주 (thr=1.0 → w>=h)."""
    w, h = img.size
    return (w / max(h, 1)) >= thr

def _parse_and_clean(raw: str):
    """모델 출력(JSON/깨짐 가능)을 리스트[dict]로 정규화."""
    cleaner = OutputCleaner()
    parsed = None
    try:
        if isinstance(raw, str) and raw.strip().startswith('['):
            parsed = json.loads(raw)
    except Exception:
        parsed = None
    cleaned = cleaner.clean_model_output(parsed if isinstance(parsed, list) else raw)
    return cleaned if isinstance(cleaned, list) else []

def _elements_to_cells_in_orig(
    elements: List[dict],
    orig_size: Tuple[int, int],
    proc_size: Tuple[int, int],
    category_map: Dict[str, str],
    scale_tag: str,               # 'hi' or 'lo'
) -> List[Cell]:
    """클린된 elements를 원본 좌표 Cell로 변환. 리스트 순서를 model-order로 보존."""
    ow, oh = orig_size
    pw, ph = proc_size
    sx = ow / max(pw, 1)
    sy = oh / max(ph, 1)

    cells: List[Cell] = []
    for idx, el in enumerate(elements):
        raw_cat = str(el.get("category", "")).strip().lower()
        mapped = category_map.get(raw_cat)
        if not mapped:
            continue
        bbox = el.get("bbox", [])
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = bbox
        xb = [int(round(x1 * sx)), int(round(y1 * sy)),
              int(round(x2 * sx)), int(round(y2 * sy))]
        if xb[2] <= xb[0] or xb[3] <= xb[1]:
            continue
        cells.append(
            Cell(
                category=mapped,
                bbox=xb,
                order_hi=idx if scale_tag == 'hi' else None,
                order_lo=idx if scale_tag == 'lo' else None,
                source=scale_tag,
                confidence_score=0.85 if scale_tag == 'hi' else 0.78,
            )
        )
    return cells

def _iou_xyxy(a: List[int], b: List[int]) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-6, (areaA + areaB - inter))

def _merge_hi_lo_by_category(
    cells_hi: List[Cell],
    cells_lo: List[Cell],
    iou_thr: float = 0.6
) -> List[Cell]:
    """카테고리별로 hi/lo를 병합. 같은 GT를 가리키는 경우 하나로 합치고 source='both'."""
    from collections import defaultdict
    by_cat = defaultdict(lambda: {'hi': [], 'lo': []})
    for c in cells_hi: by_cat[c.category]['hi'].append(c)
    for c in cells_lo: by_cat[c.category]['lo'].append(c)

    merged: List[Cell] = []
    for cat, groups in by_cat.items():
        his = groups['hi']; los = groups['lo']
        used_lo = [False]*len(los)

        # 1) hi 중심으로 매칭하여 anchor 생성
        for h in his:
            best_j, best_iou = -1, 0.0
            for j, l in enumerate(los):
                if used_lo[j]: continue
                iou = _iou_xyxy(h.bbox, l.bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j != -1 and best_iou >= iou_thr:
                used_lo[best_j] = True
                l = los[best_j]
                # 대표 bbox는 hi 유지(정밀), order는 평균 기반 키 생성
                merged.append(
                    Cell(
                        category=cat,
                        bbox=h.bbox,
                        order_hi=h.order_hi,
                        order_lo=l.order_lo,
                        source='both',
                        confidence_score=0.95,  # 합의 ↑
                    )
                )
            else:
                merged.append(h)  # hi-only

        # 2) 남은 lo-only 추가
        for j, l in enumerate(los):
            if not used_lo[j]:
                merged.append(l)

    return merged

def _order_key(cell: Cell) -> float:
    """
    모델 순서를 최대한 보존하여 정렬 키 생성.
    - both: (order_hi + order_lo)/2
    - hi-only: order_hi + 0.25
    - lo-only: order_lo + 0.65
    좌표가 비슷할 때 미세 충돌을 피하려는 가중(경험적).
    """
    if cell.source == 'both' and cell.order_hi is not None and cell.order_lo is not None:
        return (cell.order_hi + cell.order_lo) / 2.0
    if cell.source == 'hi' and cell.order_hi is not None:
        return cell.order_hi + 0.25
    if cell.source == 'lo' and cell.order_lo is not None:
        return cell.order_lo + 0.65
    # 안전장치: 없으면 좌상단 기준
    b = cell.bbox
    return b[1] * 1e4 + b[0]

def assign_final_order(merged_cells: List[Cell]) -> List[Cell]:
    """
    최종 reading order를 0부터 연속으로 부여.
    모델의 order 신호(_order_key) + 좌표 기준 정렬을 사용.
    """
    sorted_cells = sorted(merged_cells, key=_order_key)
    for i, c in enumerate(sorted_cells):
        c.order_final = i  # Cell 객체에 최종 order 속성 추가
    return sorted_cells

def infer_layout_once(
    pil_img: Image.Image,
    model,
    processor,
    device: str,
    max_pixels: int,
    category_map: Dict[str, str],
) -> Tuple[List[dict], Tuple[int,int]]:
    """
    한 스케일에서 레이아웃 1회 추론.
    반환: (elements(list[dict]), processed_image_size)
    """
    processed = fetch_image(pil_img, min_pixels=MIN_PIXELS, max_pixels=max_pixels)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": processed},
            {"type": "text", "text":
                "Please output the layout information from this PDF image, including each layout's bbox and its category. "
                "The bbox should be in the format [x1, y1, x2, y2]. "
                "The layout categories for the PDF document include "
                "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. "
                "Do not output the corresponding text. The layout result should be in JSON format."
            },
        ],
    }]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt")
    if device.startswith("cuda"):
        inputs = inputs.to(device, dtype=torch.float16)
    else:
        inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    gen_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    response = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    elements = _parse_and_clean(response)
    return elements, processed.size

def box_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def overlap_ratio(a, b):
    """a가 b를 포함하는 비율 (a∩b / b 영역)."""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
    area_b = box_area(b)
    return inter / max(area_b, 1)

def refined_global_nms_v2(cells, iou_thr=0.6, include_thr=0.8):
    """
    - 같은 카테고리: IoU > iou_thr → 작은 박스 제거
    - 다른 카테고리(text vs image): 겹치면 image 우선
    - 포함비율 > include_thr → 작은 박스 제거
    """
    keep = []
    used = [False]*len(cells)
    # 면적 기준 정렬 (작은 박스 먼저 살림)
    cells_sorted = sorted(cells, key=lambda c: box_area(c.bbox))

    for i,a in enumerate(cells_sorted):
        if used[i]: continue
        keep.append(a)
        for j in range(i+1, len(cells_sorted)):
            if used[j]: continue
            b = cells_sorted[j]
            iou = _iou_xyxy(a.bbox, b.bbox)
            incl_ab = overlap_ratio(a.bbox, b.bbox)  # a가 b 포함
            incl_ba = overlap_ratio(b.bbox, a.bbox)  # b가 a 포함

            # 같은 카테고리
            if a.category == b.category and iou > iou_thr:
                used[j] = True

            # 포함 관계 강하면 작은 쪽 제거
            elif incl_ab > include_thr or incl_ba > include_thr:
                if box_area(a.bbox) < box_area(b.bbox):
                    used[j] = True  # 큰 박스 제거
                else:
                    used[i] = True

            # cross category (image vs text)
            elif iou > iou_thr:
                if a.category == "image" and b.category in ["text","title","subtitle"]:
                    used[j] = True
                elif b.category == "image" and a.category in ["text","title","subtitle"]:
                    used[i] = True
    return keep

def filter_decorations(cells, page_size, area_thr=0.01, corner_thr=0.09):
    """
    작은 장식(페이지 번호, 로고 등)을 필터링.
    - category == "image"인 경우만 검사
    - 전체 면적 대비 area_thr 이하
    - 좌상단 또는 우상단 corner_thr 비율 안에 위치한 경우 제거
    """
    W, H = page_size
    keep = []
    for c in cells:
        if c.category == "image":
            w = c.bbox[2] - c.bbox[0]
            h = c.bbox[3] - c.bbox[1]
            area = max(1, w * h)
            ratio = area / max(1, W * H)

            # 작은 이미지일 때만 후보
            if ratio < area_thr:
                # 좌상단
                if c.bbox[0] < W * corner_thr and c.bbox[1] < H * corner_thr:
                    continue
                # 우상단
                if c.bbox[2] > W * (1 - corner_thr) and c.bbox[1] < H * corner_thr:
                    continue
        keep.append(c)
    return keep


def tta_layout(
    pil_img: Image.Image,
    model,
    processor,
    device: str,
    category_map: Dict[str, str],
    hi_pixels: int = 1_228_800,
    lo_pixels: int = 921_600,
    landscape_thr: float = 1.0,
    iou_thr_merge: float = 0.6,
    iou_thr_global: float = 0.7,
) -> List[Cell]:
    """
    세로형: hi 한 번
    가로형: hi + lo 두 번 (단, 전체 20회까지만 TTA)
    """
    global _tta_landscape_count

    is_land = is_landscape(pil_img, thr=landscape_thr)

    # hi
    el_hi, size_hi = infer_layout_once(pil_img, model, processor, device, hi_pixels, category_map)
    cells_hi = _elements_to_cells_in_orig(el_hi, pil_img.size, size_hi, category_map, 'hi')

    if not is_land:
        return cells_hi  # 세로형 → hi만

    # 가로형인데 TTA 횟수 제한 초과 → hi만
    if _tta_landscape_count >= TTA_LANDSCAPE_LIMIT:
        print("⚠️ 가로형 TTA 제한 초과 → hi만 수행")
        return cells_hi

    # 가로형이고 아직 카운트 여유 있음 → hi+lo
    _tta_landscape_count += 1
    el_lo, size_lo = infer_layout_once(pil_img, model, processor, device, lo_pixels, category_map)
    cells_lo = _elements_to_cells_in_orig(el_lo, pil_img.size, size_lo, category_map, 'lo')

    # 병합 + NMS
    merged = _merge_hi_lo_by_category(cells_hi, cells_lo, iou_thr=iou_thr_merge)
    cleaned = refined_global_nms_v2(merged, iou_thr=0.6, include_thr=0.8)
    return cleaned