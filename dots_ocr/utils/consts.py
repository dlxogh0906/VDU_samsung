# dots_ocr/utils/consts.py
# Constants for DotsOCR processing

# Image processing constants
IMAGE_FACTOR = 28  # Factor for image dimension rounding
MIN_PIXELS = 3136  # Minimum number of pixels (56x56)
MAX_PIXELS = 1228800 # Maximum number of pixels (to control token length) #2048000 ＃600000 #921600 ＃ 1048576 #1228800 #1600000 #2048000 3458560 

# Default DPI for document conversion
DEFAULT_DPI = 200

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
SUPPORTED_DOCUMENT_EXTENSIONS = ['.pdf', '.pptx']
SUPPORTED_ALL_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_DOCUMENT_EXTENSIONS

# Category mappings for competition
COMPETITION_CATEGORIES = {
    'title': 'title',
    'subtitle': 'subtitle', 
    'text': 'text',
    'image': 'image',
    'table': 'table',
    'equation': 'equation'
}

# Layout categories from DotsOCR model
LAYOUT_CATEGORIES = [
    'Caption', 'Footnote', 'Formula', 'List-item', 
    'Page-footer', 'Page-header', 'Picture', 
    'Section-header', 'Table', 'Text', 'Title'
]