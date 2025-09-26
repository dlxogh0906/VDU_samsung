# Visually-rich Document Understanding : 2025 Samsung AI Challenge
> 데이콘 x 삼성전자 AI센터
---

## [주제]

인간이 문서를 통해 표현하고자 한 의미, 강조, 구조, 의도를 해석할 수 있는 AI 모델 개발

## 📁 프로젝트 구조
<pre>
├── data/ # 데이터 샘플
├── dots_ocr/ # utils 함수 (JSON 후처리 등)
│ ├── utils/
│ │ ├── pycache/
│ │ ├── init.py
│ │ ├── consts.py
│ │ ├── doc_utils.py
│ │ ├── format_transformer.py
│ │ ├── image_utils.py
│ │ ├── layout_utils.py
│ │ ├── orientation_utils.py
│ │ ├── output_cleaner.py
│ │ ├── postprocess.py
│ │ └── tta_utils.py
│ └── init.py
│
├── model/ # 모델 정의 및 학습 모듈
│ ├── dots.ocr
| ├── Callisto-OCR3-2B-Instruct
│
├── main_callisto_0910.py # EDA 및 실험용 스크립트
├── requirements.txt # 의존성 패키지
└── README.md # 프로젝트 소개 문서
</pre>
---
## 🏆 대회 결과

| 항목             | 내용                             |
|------------------|----------------------------------|
| 🥇 최종 순위     |  **3위 / 246팀**                     |
| 📈 Public Score  | 0.4391                          |





