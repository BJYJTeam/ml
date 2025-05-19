# BJYJ-ML (Django AI API Server)

## 📡 사용 가능한 API

- `POST /api/answer`            → 일반 AI 답변 생성 (Gemma3 API 연동)
- `POST /api/doctor-answer`     → 의사 스타일 답변 생성 (Gemma3 API 연동)
- `POST /api/extract-keywords`  → 질문 제목/내용으로부터 키워드 추출
- `POST /api/similar-questions` → 질문과 유사한 기존 질문 반환
- `GET  /api/faqs`              → FAQ 질문 목록 반환

---

## 🚀 실행 방법 (로컬 개발용)

```bash
git clone https://github.com/YOUR_TEAM/bjyj-ml.git
cd bjyj-ml

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python manage.py runserver
```

---

## 🐳 Docker로 실행

```bash
docker compose up --build
```

--- 

## 🧠 모델 개발자 가이드

```bash
pip freeze > requirements.txt
```

### 여기만 작업하면 됩니다:

1. `inference.py`에서 다음 항목 수정:
   - 질문 데이터프레임 로딩 (`df = pd.read_csv("qa_data.csv")`)
   - 질문 전처리 함수 (`preprocess`)
   - 임베딩 모델 로딩 및 `question_embeddings` 생성
   - 주요 추론 함수:
     - `ai_answer()`
     - `doctor_answer()`
     - `extract_keywords_from_model()`
     - `find_similar_questions()`

2. 필요한 추가 데이터 파일은 루트 폴더에 자유롭게 추가
   - 예: `qa_data.json`, `naver_qna_data.csv` 등