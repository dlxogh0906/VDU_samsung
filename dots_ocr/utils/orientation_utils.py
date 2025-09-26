# dots_ocr/utils/orientation_utils.py
from PIL import Image

def rotate_if_landscape(img: Image.Image):
    """가로형이면 90도 회전해서 반환, 아니면 그대로 반환"""
    w, h = img.size
    if w > h:
        return img.rotate(90, expand=True), True
    return img, False

def restore_bbox_from_rotated(bbox, orig_size):
    """회전된 좌표계를 원본 landscape 좌표계로 변환"""
    W, H = orig_size  # 원본 (landscape)
    x1, y1, x2, y2 = bbox
    return [
        H - y2,  # new_x1
        x1,      # new_y1
        H - y1,  # new_x2
        x2       # new_y2
    ]