# dots_ocr/utils/postprocess.py
from typing import List, Tuple, Dict

BBox = List[int]                     # [x1, y1, x2, y2]
Row  = Dict[str, object]             # submission row dict
HeaderCand = Tuple[BBox, float]      # (bbox, confidence)

def _area(b: BBox) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = _area(a) + _area(b) - inter + 1e-6
    return inter / ua

def _parse_bbox_str(bstr: str) -> BBox:
    return [int(v) for v in bstr.split(",")]

def _fmt_bbox(b: BBox) -> str:
    return f"{b[0]}, {b[1]}, {b[2]}, {b[3]}"

def promote_headers_to_title(
    submission_rows: List[Row],
    header_cands: List[HeaderCand],
    target_size: Tuple[int, int],
    doc_id: str,
    *,
    top_ratio: float = 0.20,         # y1 < 20%*H
    min_w_ratio: float = 0.20,       # w/W ≥ 20%
    min_area_ratio: float = 0.01,    # area/(W*H) ≥ 1%
    min_h_ratio: float = 0.015,      # 1.5% ≤ h/H
    max_h_ratio: float = 0.12,       # h/H ≤ 12%
    iou_thresh: float = 0.5,         # 기존 title과 겹치지 않게
    default_conf: float = 0.85,
) -> List[Row]:
    """
    page-header/page_footer 후보(header_cands)를 Title로 0~1개 승격하여
    submission_rows에 추가한다. (조건 안 맞으면 무동작)

    반환값: 수정된 submission_rows (in-place 편의로 동일 객체를 반환)
    """
    if not header_cands:
        return submission_rows

    W, H = target_size

    def pass_geom(b: BBox) -> bool:
        x1, y1, x2, y2 = b
        w, h = (x2 - x1), (y2 - y1)
        if H <= 0 or W <= 0:
            return False
        top_ok    = (y1 < top_ratio * H)
        width_ok  = (w / W >= min_w_ratio)
        h_ratio   = h / H
        area_ok   = (w / W) * h_ratio >= min_area_ratio
        height_ok = (min_h_ratio <= h_ratio <= max_h_ratio)
        return top_ok and width_ok and height_ok and area_ok

    # 조건 통과 후보만 남기고 면적 큰 순으로 정렬
    valid = [(b, c) for (b, c) in header_cands if pass_geom(b)]
    valid.sort(key=lambda t: _area(t[0]), reverse=True)
    if not valid:
        return submission_rows

    # 이미 감지된 title들과 IoU 체크
    title_boxes = [_parse_bbox_str(r["bbox"]) for r in submission_rows
                   if r.get("category_type") == "title"]

    chosen = None
    if title_boxes:
        for b, c in valid:
            if all(_iou(b, tb) <= iou_thresh for tb in title_boxes):
                chosen = (b, c)
                break
    else:
        chosen = valid[0]

    if chosen is None:
        return submission_rows

    b, c = chosen
    submission_rows.append({
        "ID": doc_id,
        "category_type": "title",
        "confidence_score": float(c) if c is not None else default_conf,
        "order": 0,            # 아래에서 reorder로 다시 정렬
        "text": "",
        "bbox": _fmt_bbox(b),
    })
    return submission_rows

def reorder_by_reading_order(submission_rows: List[Row]) -> None:
    """
    y1 오름차순, 동률 시 x1로 정렬하고 order를 0..N-1로 재부여 (in-place)
    """
    def key_order(r: Row):
        x1, y1, _, _ = _parse_bbox_str(r["bbox"])
        return (y1, x1)

    submission_rows.sort(key=key_order)
    for i, r in enumerate(submission_rows):
        r["order"] = i