# 온누리 통증의학과 온라인 상담 자동화 시스템 (BJYJ-ML)

본 레포지토리는 병원 Q&A 게시판의 반복 문의를 줄이고 의료진의 답변 부담을 완화하기 위해 설계된 **데이터 기반 의료 상담 자동화 API 서버**입니다. 질문 분석부터 키워드 추출, 유사 질문 검색, FAQ 생성, AI 초안 생성, 이미지 추천까지 한 파이프라인에서 제공합니다.

## 핵심 강점

- **의미 기반 검색/중복 제거**: SBERT 임베딩 + HDBSCAN 클러스터링으로 유사 질문을 통합하고 대표 질문을 선정합니다.
- **LLM 중심 자동화**: Gemma3 기반 프롬프트로 의료 맥락을 반영한 키워드 추출과 답변 생성을 수행합니다.
- **의료진 보조 워크플로우**: AI 답변에 대한 사용자 피드백이 들어오면 의료진 초안을 자동 생성해 후처리를 돕습니다.
- **시각자료 추천**: 질문 문맥과 이미지 메타데이터 임베딩의 코사인 유사도를 활용해 참고 이미지를 추천합니다.
- **운영 친화 API**: DRF 기반 REST API와 Swagger 문서화로 서비스 연동이 쉽습니다.

## 시스템 개요

1. **키워드/태그 추출**: 질문 내용을 LLM으로 분석해 질환/증상/행동 기반 태그를 생성합니다.
2. **유사 질문 검색 및 FAQ 자동 생성**: SBERT 임베딩 후 HDBSCAN으로 군집화해 대표 질문을 추출하고 FAQ를 요약 생성합니다.
3. **AI 자동 답변 생성**: 유사 질문 3건과 전문의 답변을 컨텍스트로 Gemma3가 답변을 작성합니다.
4. **의료진 초안 지원**: 사용자 피드백이 있을 경우 AI가 전문의 스타일 초안을 생성합니다.
5. **의료 이미지 추천**: 의료 메타데이터 기반 유사 이미지 5건을 추천합니다.

## 기술 스택

- **Backend**: Django, DRF, drf-yasg(Swagger)
- **NLP/ML**: SentenceTransformer(SBERT), HDBSCAN, Gemma3 LLM, cosine similarity
- **Data**: Q&A 데이터, 전문의 답변 데이터, 의료 이미지 메타데이터

## 프로젝트 구조

- `ai_server/`: Django 설정 및 URL 라우팅
- `ai_app/`: API 뷰 + 추론 로직
- `docs/`: 질문/댓글 데이터 및 메타데이터
- `media/`: 이미지 에셋
- `test_image_recommendation.py`: 이미지 추천 간단 테스트

## 제공 API

- `POST /api/answer`            → 일반 AI 답변 생성
- `POST /api/doctor-answer`     → 의료진 스타일 초안 생성
- `POST /api/extract-keywords`  → 질문 키워드 추출
- `POST /api/similar-questions` → 유사 질문 ID 검색
- `GET  /api/faqs`              → FAQ 자동 생성 결과
- `POST /api/recommend-images`  → 질문 기반 이미지 추천

## 실행 방법 (로컬)

```bash
git clone https://github.com/YOUR_TEAM/bjyj-ml.git
cd bjyj-ml

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python manage.py runserver
```

## Docker 실행

```bash
docker compose up --build
```

## 환경변수

`.env`에 아래 값을 설정하세요.

```bash
DJANGO_SECRET_KEY=...
GEMMA_API_TOKEN=...
API_URL=...
```

## 모델/데이터 구성

- 추론 로직: `ai_app/inference.py`
- 질문/답변 데이터: `docs/qa_DB_tag_json.csv`, `docs/post_comments.json`
- 이미지 메타데이터: `docs/medical_metadata.json`

## 간단 테스트

```bash
python test_image_recommendation.py
```
