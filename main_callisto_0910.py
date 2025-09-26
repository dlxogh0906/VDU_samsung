import os
import json
import subprocess
from pathlib import Path
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils.image_utils import fetch_image
from dots_ocr.utils.doc_utils import load_images_from_pdf
from dots_ocr.utils.format_transformer import get_formula_in_markdown, clean_text, has_latex_markdown
from dots_ocr.utils.output_cleaner import OutputCleaner
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.layout_utils import draw_layout_on_image  # 시각화는 안 씀
import time
import numpy as np
from transformers import Qwen2VLForConditionalGeneration
import re
import pytesseract
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = os.path.join("model", "dots.ocr", "weights", "DotsOCR")  # 레이아웃 감지용
CALLISTO_MODEL_PATH = os.path.join("model", "Callisto-OCR3-2B-Instruct")  # Callisto OCR 로컬 모델
SUBMISSION_PATH = os.path.join("output", "submission.csv")
TEMP_DIR = "./temp_images"
DEBUG_JSON_DIR = "./debug_json"

USE_VRAM_LIMIT = False   # True → VRAM 제한 모드 (A6000에서 T4 시뮬레이션)
                        # False → 일반 모드 (T4 같은 실제 16GB 환경)

def run_tesseract(image_pil):
    """테서렉트 OCR 실행 (kor+eng)"""
    padded = Image.new("RGB", (image_pil.width + 20, image_pil.height + 20), "white")
    padded.paste(image_pil, (10, 10))
    text = pytesseract.image_to_string(
        padded, lang="kor+eng", config="--psm 6"
    )
    return text.strip()

def preprocess_for_layout(image, min_size=1000):
    """레이아웃 탐지 전 이미지 전처리 (선명도, 해상도, 노이즈)"""
    try:
        # 해상도 보정 (너무 작으면 확대)
        w, h = image.size
        if max(w, h) < min_size:
            scale = min_size / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # 선명도 강화
        image = ImageEnhance.Contrast(image).enhance(1.2)
        image = ImageEnhance.Sharpness(image).enhance(1.2)

        # 노이즈 제거 (필터)
        image = image.filter(ImageFilter.MedianFilter(size=3))

        return image
    except Exception as e:
        print(f"⚠️ 레이아웃 전처리 실패: {e}")
        return image

def is_number_pair(text: str) -> bool:
    """
    (숫자),(숫자) 패턴인지 확인
    - 천 단위 콤마(예: 999,999)도 허용
    """
    return bool(re.fullmatch(r"^\(\d{1,3}(?:,\d{3})*\),\(\d{1,3}(?:,\d{3})*\)$", text.strip()))

def has_chinese(text: str, threshold: float = 0.2) -> bool:
    """한자가 일정 비율 이상 포함되어 있으면 True"""
    if not text:
        return False
    total = len(text)
    chinese = sum('\u4e00' <= ch <= '\u9fff' for ch in text)
    return (chinese / total) >= threshold


def limit_vram(max_gb=16, device=0):
    """PyTorch VRAM fraction 제한 (예약 포함)"""
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = max_gb * (1024**3) / total
    fraction = min(1.0, fraction)  # 🚧 fraction은 0~1만 허용
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    print(f"🚧 VRAM 제한 설정: {max_gb}GB (GPU 전체 {total/1024**3:.1f}GB 중 {fraction*100:.1f}%)")

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 레이아웃 감지용 모델 (기존 DotsOCR)
    if device.startswith("cuda"):
        if USE_VRAM_LIMIT:
            # ★ VRAM 제한 모드 (A6000에서 T4 시뮬레이션)
            limit_vram(16, 0)
            layout_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="sdpa",
                max_memory={0: "8GiB", "cpu": "16GiB"}  # 메모리 분배 조정
            )
        else:
            # ★ 일반 모드 (T4 같은 실제 16GB 환경)
            layout_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="sdpa",
                local_files_only=True,
            )
    else:
        layout_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )

    layout_processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

    # Callisto OCR 모델 로딩
    callisto_model = Qwen2VLForConditionalGeneration.from_pretrained(
        CALLISTO_MODEL_PATH,
        torch_dtype=torch.float16,  # float16으로 메모리 절약
        device_map="cuda" if device.startswith("cuda") else "cpu",
        local_files_only=True,
        low_cpu_mem_usage=True,  # CPU 메모리 사용량 최적화
        attn_implementation="sdpa"  # 메모리 효율적인 attention
    )
    
    # Tokenizer와 Processor를 별도로 로드하고 padding_side 설정
    callisto_tokenizer = AutoTokenizer.from_pretrained(CALLISTO_MODEL_PATH, local_files_only=True)
    callisto_tokenizer.padding_side = 'left'  # decoder-only 모델에 적합
    
    callisto_processor = AutoProcessor.from_pretrained(CALLISTO_MODEL_PATH, local_files_only=True)
    callisto_processor.tokenizer.padding_side = 'left'  # processor 내부 tokenizer도 설정
    
    return layout_model, layout_processor, callisto_model, callisto_processor, device

# ------------------------------------------------------
# Step 2: 데이터 로드 함수
# ------------------------------------------------------
def load_data():
    data_path = os.path.join("data", "test.csv")
    df = pd.read_csv(data_path)
    return df

# ------------------------------------------------------
# Step 3: 문서 처리 클래스 (추론 로직 포함)
# ------------------------------------------------------
class CompetitionProcessor:
    def __init__(self):
        self.category_map = {
            "title": "title",
            "section-header": "subtitle",
            "text": "text",
            "picture": "image",
            "table": "table",
            "formula": "equation",
            "list-item": "text",
            "caption": "text",
        }
        self.cleaner = OutputCleaner()

    def convert_to_images(self, input_path, temp_dir, dpi=200):
        ext = Path(input_path).suffix.lower()
        os.makedirs(temp_dir, exist_ok=True)

        if ext == ".pdf":
            return load_images_from_pdf(input_path, dpi=dpi)
        elif ext == ".pptx":
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, input_path],
                check=True,
            )
            pdf_path = os.path.join(temp_dir, Path(input_path).with_suffix(".pdf").name)
            return load_images_from_pdf(pdf_path, dpi=dpi)
        elif ext in [".jpg", ".jpeg", ".png"]:
            image = Image.open(input_path)
            return [image.convert("RGB")]
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")
        
    def create_prompt(self):
        return """Please output the layout information from this PDF image, 
including each layout's bbox and its category. 
The bbox should be in the format [x1, y1, x2, y2]. 
The layout categories for the PDF document include 
['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. 
Do not output the corresponding text. 
The layout result should be in JSON format."""

    def remove_bullet_symbols(self, text):
        """글머리 기호 및 불필요한 특수문자 제거"""
        if not text:
            return ""
        
        # 제거할 기호 집합
        bullet_symbols = "▶▷◀◁►◄⊙◇◆■□▲△▼▽★☆※▲"
        
        # 문자열 시작 부분의 기호 + 공백 제거
        text = re.sub(rf'^[\s]*[{re.escape(bullet_symbols)}]+[\s]*', '', text)
        
        # 줄 시작 부분의 기호 + 공백 제거 (여러 줄 처리)
        text = re.sub(rf'^[\s]*[{re.escape(bullet_symbols)}]+[\s]*', '', text, flags=re.MULTILINE)
        
        # 텍스트 전체에서 등장하는 기호 모두 제거
        text = re.sub(rf'[{re.escape(bullet_symbols)}]', '', text)
        
        return text.strip()

    def process_text_by_category(self, text, category):
        if not text:
            return ""
        
        text = clean_text(text)
        
        # ✅ 글머리 기호 제거 추가
        text = self.remove_bullet_symbols(text)

        # ✅ 일반 텍스트 계열(title, subtitle, text)만 변환 적용
        if category in ["title", "subtitle", "text"]:
            # \( ... \) → $...$
            text = re.sub(r'\\\((.*?)\\\)', lambda m: f"${m.group(1).strip()}$", text)
        
        return text.strip()
    def validate_and_scale_bbox(self, bbox, current_size, target_size):
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(x) for x in bbox]
            current_w, current_h = current_size
            target_w, target_h = target_size
            if x1 < 0 or y1 < 0 or x2 > current_w or y2 > current_h:
                return None
            if x2 <= x1 or y2 <= y1:
                return None
            scale_x = target_w / current_w
            scale_y = target_h / current_h
            return [
                max(0, min(target_w - 1, int(round(x1 * scale_x)))),
                max(0, min(target_h - 1, int(round(y1 * scale_y)))),
                max(1, min(target_w, int(round(x2 * scale_x)))),
                max(1, min(target_h, int(round(y2 * scale_y)))),
            ]
        except Exception:
            return None

    def parse_model_output(self, raw_output):
        try:
            if raw_output.strip().startswith("["):
                parsed = json.loads(raw_output)
            else:
                parsed = None
        except:
            parsed = None
        cleaned = self.cleaner.clean_model_output(parsed if parsed is not None else raw_output)
        return cleaned if isinstance(cleaned, list) else []

    def preprocess_image_for_callisto(self, image, target_size=512):
        """Callisto-OCR3-2B용 OCR 전처리 (bbox 단위)"""
        try:
            
            
            # 1. 패딩 추가 (흰 배경)
            width, height = image.size
            padding = max(20, min(width, height) // 10)
            padded_image = Image.new('RGB', (width + 2*padding, height + 2*padding), color='white')
            padded_image.paste(image, (padding, padding))
            image = padded_image

            # 2. 크기 보정 (너무 작은 bbox는 확대)
            w, h = image.size
            min_size = 300
            if w < min_size or h < min_size:
                scale = max(min_size / w, min_size / h)
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            # 3. 대비/선명도 강화 (Callisto-friendly)
            image = ImageEnhance.Contrast(image).enhance(1.3)
            image = ImageEnhance.Sharpness(image).enhance(1.3)

            # 4. 최종 크기 정규화 (aspect 유지 + padding)
            w, h = image.size
            scale = target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)

            final_image = Image.new("RGB", (target_size, target_size), color="white")
            final_image.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))

            return final_image

        except Exception as e:
            print(f"⚠️ Callisto 전처리 실패: {e}")
            return image


    def extract_text_batch_with_callisto_ocr(self, image_pil, bbox_list, callisto_model, callisto_processor, device, category_list=None):
        """Callisto OCR을 사용하여 배치 OCR 수행"""
        if not bbox_list:
            return []
        
        # category_list는 현재 미사용 (향후 확장용)
        _ = category_list
            
        batch_messages = []
        cropped_images = []
        
        # 배치로 이미지 전처리
        for i, bbox in enumerate(bbox_list):
            x1, y1, x2, y2 = bbox
            w, h = image_pil.width, image_pil.height

            # bbox 절대값 5픽셀 확장
            dx = 5
            dy = 5

            x1e = max(0, x1 - dx)
            y1e = max(0, y1 - dy)
            x2e = min(w, x2 + dx)
            y2e = min(h, y2 + dy)

            # 너무 작은 영역은 스킵
            if (x2e - x1e) < 10 or (y2e - y1e) < 10:
                cropped_images.append(None)
                continue

            cropped = image_pil.crop((x1e, y1e, x2e, y2e)).convert("RGB")
            
            # OCR 품질 향상을 위한 이미지 전처리
            cropped = self.preprocess_image_for_callisto(cropped)
            cropped_images.append(cropped)
            
            # OCR 프롬프트 (Callisto에 최적화)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": cropped},
                        {"type": "text", "text": "Read and transcribe all visible Korean, English text, numbers, and math expressions. Return only the text, no explanations."}, #Extract the text content from this image.
                    ], # Read and transcribe all visible Korean, English text, numbers, and math expressions. Return only the text, no explanations.
                }
            ]
            batch_messages.append(messages)
        
        # 배치 처리
        results = []
        batch_size = 6  # Callisto 모델에 맞게 배치 크기 조정
        
        try:
            for i in range(0, len(batch_messages), batch_size):
                batch = batch_messages[i:i+batch_size]
                batch_crops = cropped_images[i:i+batch_size]
                
                # None인 이미지는 스킵
                valid_batch = [(msg, crop) for msg, crop in zip(batch, batch_crops) if crop is not None]
                if not valid_batch:
                    results.extend([""] * len(batch))
                    continue
                
                # 배치 처리
                texts = []
                images_batch = []
                
                for messages, _ in valid_batch:
                    text = callisto_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    texts.append(text)
                    image_inputs, _ = process_vision_info(messages)
                    images_batch.extend(image_inputs)
                
                if texts and images_batch:
                    inputs = callisto_processor(
                        text=texts,
                        images=images_batch,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(device)
                    
                    with torch.no_grad():
                        # Callisto에 최적화된 생성 설정
                        generated_ids = callisto_model.generate(
                            **inputs, 
                            max_new_tokens=1024,  # 더 긴 텍스트 생성 허용
                            do_sample=True,
                            temperature=0.01,  # 낮은 온도로 일관성 향상
                            top_p=0.9,
                            num_beams=1,
                            pad_token_id=callisto_processor.tokenizer.pad_token_id,
                            use_cache=True
                        )
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_texts = callisto_processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        # 메모리 정리
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
                    
                    # 결과 정리
                    valid_idx = 0
                    for crop in batch_crops:
                        if crop is not None:
                            text = output_texts[valid_idx] if valid_idx < len(output_texts) else ""
                            results.append(clean_text(text) if text else "")
                            valid_idx += 1
                        else:
                            results.append("")
                else:
                    results.extend([""] * len(batch))
                    
        except Exception as e:
            print(f"배치 OCR 오류: {str(e)[:120]}")
            results.extend([""] * (len(bbox_list) - len(results)))
        
        return results

    def extract_text_with_callisto_ocr(self, image_pil, bbox, callisto_model, callisto_processor, device, category=None):
        """단일 OCR 수행 (기존 호환성 유지)"""
        results = self.extract_text_batch_with_callisto_ocr(image_pil, [bbox], callisto_model, callisto_processor, device, [category])
        return results[0] if results else ""
            
    def process_single_image(self, doc_id, image, layout_model, layout_processor, callisto_model, callisto_processor, device, target_size):
        # ✅ 레이아웃 탐지 전처리 추가
        image = preprocess_for_layout(image)

        processed_image = fetch_image(image, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": processed_image}, {"type": "text", "text": self.create_prompt()}],
            }
        ]
        text_input = layout_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = layout_processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt")
        if device.startswith("cuda"):
            inputs = inputs.to(device, dtype=torch.float16)
        else:
            inputs = inputs.to(device)

        with torch.no_grad():
            output_ids = layout_model.generate(
                **inputs, 
                max_new_tokens=2048,
                do_sample=False,
                num_beams=1,
                pad_token_id=layout_processor.tokenizer.pad_token_id,
                use_cache=True
             )
        if device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"💾 GPU 메모리 [Doc {doc_id}] → allocated: {allocated:.1f} MB, reserved: {reserved:.1f} MB")
        generated_ids = [output_ids[i][len(inputs.input_ids[i]) :] for i in range(len(output_ids))]
        response = layout_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print(response)
        elements = self.parse_model_output(response)

        submission_rows = []
        ocr_needed_elements = []  # OCR 필요한 요소들 저장
        model_w, model_h = processed_image.size
        order = 0
        
        # 1단계: 모든 레이아웃 요소 처리 및 OCR 대상 수집
        for element in elements:
            raw_category = str(element.get("category", "")).strip().lower()
            mapped_category = self.category_map.get(raw_category)
            if not mapped_category:
                continue

            bbox = element.get("bbox", [])
            scaled_bbox = self.validate_and_scale_bbox(bbox, (model_w, model_h), target_size)
            if not scaled_bbox:
                continue

            orig_bbox = self.validate_and_scale_bbox(bbox, (model_w, model_h), image.size)
            # OCR 수행 여부 결정
            if mapped_category in ["title", "subtitle", "text"]:
                ocr_needed_elements.append((len(submission_rows), element, orig_bbox, mapped_category))
                text_content = ""  # 일단 빈 텍스트로 초기화
            else:
                text_content = ""  # OCR 미수행

            submission_rows.append(
                {
                    "ID": doc_id,
                    "category_type": mapped_category,
                    "confidence_score": 0.85,
                    "order": order,
                    "text": text_content,
                    "bbox": f"{scaled_bbox [0]}, {scaled_bbox [1]}, {scaled_bbox [2]}, {scaled_bbox [3]}",
                }
            )
            order += 1
        
        # 2단계: 배치로 OCR 수행
        if ocr_needed_elements:
            bbox_list = [elem[2] for elem in ocr_needed_elements]  # orig_bbox들
            category_list = [elem[3] for elem in ocr_needed_elements]  # category들
            
            ocr_results = self.extract_text_batch_with_callisto_ocr(
                image_pil=image,
                bbox_list=bbox_list,
                callisto_model=callisto_model,
                callisto_processor=callisto_processor,
                device=device,
                category_list=category_list
            )
            
            # 3단계: OCR 결과를 submission_rows에 적용
# 3단계: OCR 결과를 submission_rows에 적용
            for i, (row_idx, element, orig_bbox, category) in enumerate(ocr_needed_elements):
                if i < len(ocr_results):
                    text_content = self.process_text_by_category(ocr_results[i], category)

                    # ✅ fallback 조건 검사
                    if not text_content or is_number_pair(text_content) or has_chinese(text_content, threshold=0.2):
                        x1, y1, x2, y2 = orig_bbox
                        crop = image.crop((x1, y1, x2, y2)).convert("RGB")

                        # 1차 fallback → Callisto 재시도 (흰색 패딩 크게 넣음)
                        padded_crop = self.preprocess_image_for_callisto(crop, target_size=512)
                        retry = self.extract_text_with_callisto_ocr(
                            padded_crop, [0, 0, padded_crop.width, padded_crop.height],
                            callisto_model, callisto_processor, device, category
                        )
                        if retry:
                            text_content = retry
                        else:
                            # 2차 fallback → Tesseract
                            text_tess = run_tesseract(crop)
                            if text_tess:
                                text_content = text_tess

                    submission_rows[row_idx]["text"] = text_content


        # 시각화 저장 (옵션)
        SAVE_VISUALIZATION = True
        if SAVE_VISUALIZATION and submission_rows:
            try:
                vis_img = image.resize(target_size)
                vis_img = draw_layout_on_image(
                    vis_img,
                    [
                        {"bbox": [int(v) for v in row["bbox"].split(",")],
                        "category": row["category_type"].capitalize()}
                        for row in submission_rows
                    ]
                )
                vis_dir = "./output/vis"
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{doc_id}.png")
                vis_img.save(vis_path)
                print(f"🖼️ 시각화 저장 완료: {vis_path}")
            except Exception as e:
                print(f"⚠️ 시각화 저장 실패 ({doc_id}): {e}")
        return submission_rows

# ------------------------------------------------------
# Step 4: predict (추론 함수)
# ------------------------------------------------------
def predict(layout_model, layout_processor, callisto_model, callisto_processor, device, df):
    processor_instance = CompetitionProcessor()
    all_submission_rows = []
    csv_dir = os.path.dirname(os.path.join("data", "test.csv"))

    for _, row in df.iterrows():
        doc_id = row["ID"]
        raw_path = row["path"]
        file_path = os.path.normpath(os.path.join(csv_dir, raw_path))
        target_size = (int(row["width"]), int(row["height"]))

        if not os.path.exists(file_path):
            print(f"⚠️ 파일 없음: {file_path}")
            continue

        try:
            images = processor_instance.convert_to_images(file_path, TEMP_DIR)
            for page_idx, img in enumerate(images):
                page_id = f"{doc_id}_p{page_idx+1}" if len(images) > 1 else doc_id
                try:
                    page_results = processor_instance.process_single_image(
                        doc_id=page_id,
                        image=img,
                        layout_model=layout_model,
                        layout_processor=layout_processor,
                        callisto_model=callisto_model,
                        callisto_processor=callisto_processor,
                        device=device,
                        target_size=target_size,
                    )
                    all_submission_rows.extend(page_results)
                except Exception as e:
                    print(f"❌ 페이지 처리 실패: {page_id} → {e}")
                finally:
                    # 메모리 정리 강화
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # 이미지 객체 메모리 해제
                    del img
                    if 'page_results' in locals():
                        del page_results
        except Exception as e:
            print(f"❌ 문서 처리 실패: {file_path} → {e}")
            continue

    return all_submission_rows

# ------------------------------------------------------
# Step 5: save_results (결과 저장)
# ------------------------------------------------------
def save_results(predictions):
    os.makedirs("output", exist_ok=True)
    submission_df = pd.DataFrame(predictions, columns=["ID", "category_type", "confidence_score", "order", "text", "bbox"])
    submission_df.to_csv(SUBMISSION_PATH, index=False, encoding="UTF-8-sig")
    print(f"제출파일 저장 완료: {SUBMISSION_PATH}, 총 항목 수 {len(submission_df)}")

# ------------------------------------------------------
# 메인 실행
# ------------------------------------------------------
if __name__ == "__main__":
    
    start_time = time.time()   # 시작 시각 기록
    
    layout_model, layout_processor, callisto_model, callisto_processor, device = load_models()
    df = load_data()
    predictions = predict(layout_model, layout_processor, callisto_model, callisto_processor, device, df)
    if predictions:
        save_results(predictions)
    else:
        print("결과가 생성되지 않았습니다.")
        
    end_time = time.time()     # 끝 시각 기록
    elapsed = end_time - start_time
    print(f"⏱️ 총 추론 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")