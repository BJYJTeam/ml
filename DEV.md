# BJYJ-ML Development Guide

이 저장소의 Django 애플리케이션은 일반 도메인 백엔드가 아니라 모델 추론과 외부 서비스 연동을 담당하는 AI/ML 서빙 레이어입니다. 사용자, 게시글 등 도메인 기능은 별도 백엔드 저장소에서 담당합니다.

README에는 서비스 구조와 핵심 AI pipeline만 유지하고, 로컬 실행·환경변수·benchmark·test 같은 개발 세부사항은 이 문서에서 관리합니다.

## Local Setup

```bash
git clone https://github.com/BJYJTeam/ml.git
cd ml

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

환경변수를 설정한 뒤 Django 기반 AI/ML 서빙 서버를 실행합니다.

```bash
set -a
source .env
set +a

python manage.py migrate
python manage.py runserver
```

기본 주소는 `http://127.0.0.1:8000/`입니다.

로컬 cache에 model이 없으면 해당 추론 기능을 처음 호출할 때 Hugging Face에서 model을 다운로드합니다.

## Docker

Compose는 AI/ML 서빙 서비스를 실행하며 외부 network `root_bjyj_network`를 사용합니다.

```bash
docker network create root_bjyj_network
docker compose up --build
```

Compose 설정에는 `PYTHONUNBUFFERED`만 기본 주입되므로 Gemma 기능을 사용하려면 필요한 secret과 endpoint를 container environment에 별도로 전달해야 합니다.

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | Production | Django secret key. `DJANGO_DEBUG=false`인 경우 필수 |
| `DJANGO_DEBUG` | Optional | 개발 mode 및 Swagger UI 활성화 |
| `DJANGO_ALLOWED_HOSTS` | Production | 쉼표로 구분한 허용 Host 목록 |
| `CORS_ALLOWED_ORIGINS` | Web Client | 쉼표로 구분한 허용 Origin 목록 |
| `GEMMA_API_TOKEN` | LLM Features | Gemma API authorization token |
| `API_URL` | LLM Features | Gemma compatible generation API endpoint |
| `GEMMA_TIMEOUT_SECONDS` | Optional | API timeout. 기본값 `15`초 |
| `GEMMA_MAX_RETRIES` | Optional | 일시적 network error의 추가 retry 횟수. 기본값 `1` |

실제 secret 값은 repository에 commit하지 않습니다.

## Runtime Behavior

- **Request Validation**: Serializer가 필수값, 공백 문자열, 제목·본문·댓글 길이를 검증하고 잘못된 요청에는 `400 Bad Request`를 반환합니다.
- **LLM Failure Handling**: 일시적인 network error만 `GEMMA_MAX_RETRIES`만큼 추가 재시도합니다. 설정 오류, HTTP 비성공 응답, malformed JSON, 빈 응답은 즉시 제어된 오류로 처리하며 API에서는 `503 Service Unavailable`를 반환합니다.
- **Lazy Loading**: retrieval model·Q&A index와 image model·metadata는 해당 기능을 처음 호출할 때 로드한 뒤 cache합니다.
- **FAQ Review Control**: FAQ batch 결과는 `pending_review`로 저장되고, `approved` 상태의 항목만 FAQ API에 공개됩니다.

## Retrieval Benchmark

과거 Q&A 174건과 validation query 44건을 이용해 retrieval model을 비교합니다.

```bash
python manage.py benchmark_retrieval_models \
  --output docs/validation/retrieval-benchmark-v1.json
```

관련 파일:

- `docs/validation/retrieval-relevance-v1.json`
- `docs/validation/retrieval-benchmark-v1.json`

현재 선택된 retrieval model은 `intfloat/multilingual-e5-small`이며 threshold는 `0.9`입니다.

### Model Selection Result

| Model | Top-1 | Top-3 | Threshold | Irrelevant Exclusion | Balanced Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` | 5.0% | 20.0% | 0.8 | 75.0% | 47.5% |
| `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | 42.5% | 67.5% | 0.7 | 100.0% | 78.75% |
| `intfloat/multilingual-e5-small` | **62.5%** | **75.0%** | **0.9** | **100.0%** | **85.0%** |

평가는 과거 Q&A 174건과 관련성 검토를 마친 validation query 44건을 사용합니다. 선택 기준은 threshold 적용 후의 Top-3 recall과 no-match specificity의 평균이며, 전체 결과는 `docs/validation/retrieval-benchmark-v1.json`에 저장합니다.

## FAQ Clustering Benchmark

representation, embedding model, HDBSCAN parameter 조합을 비교합니다.

```bash
python manage.py benchmark_faq_clustering \
  --output docs/validation/faq-clustering-benchmark-v1.json
```

현재 기본 설정:

```text
min_cluster_size=3
min_samples=2
cluster_selection_method=leaf
```

관련 파일:

- `docs/validation/faq-clustering-benchmark-v1.json`

### Configuration Selection Result

109개 설정을 비교했습니다. 비교 대상은 표현 방식, 세 embedding model, HDBSCAN의 `min_cluster_size`·`min_samples`·cluster selection 방식입니다. cluster 수를 목표로 삼지 않고, largest cluster ratio, noise ratio, cluster coherence, topic purity를 함께 평가합니다.

| 증상·진단 122건 | Representation / Configuration | Clusters | Largest Cluster | Noise |
| --- | --- | ---: | ---: | ---: |
| 기존 방식 | `all-MiniLM-L6-v2`, keyword + body, EOM, 4 / 2 | 2 | 95.9% | 0.8% |
| E5 EOM | title + keyword, EOM, 3 / 1 | 11 | 23.8% | 39.3% |
| **Selected** | **E5, title + keyword, leaf, 3 / 2** | **13** | **5.7%** | **59.0%** |

선택 설정은 큰 cluster의 과도한 점유를 완화합니다. `치료 및 시술 문의` 13건은 밀도 군집을 만들지 못해 모두 noise로 남으며, 이 경우 FAQ를 생성하지 않습니다.

## FAQ Draft Generation

FAQ draft는 API 요청 중 생성하지 않고 batch command로 생성합니다.

```bash
python manage.py generate_faq_drafts \
  --output docs/faq_drafts/faq-drafts-v1.json
```

새 draft는 기본적으로 `pending_review` 상태입니다.

- `pending_review`: 검토 대기
- `approved`: API 공개 대상
- `rejected`: 비공개
- parsing error: 비공개

## Tests

```bash
pytest
pytest --cov=ai_app --cov-report=term-missing
```

Image recommendation을 수동 확인하려면:

```bash
python test_image_recommendation.py
```

자동화 테스트에서는 외부 model download와 Gemma API 호출을 대체하여 다음 항목을 검증합니다.

- semantic retrieval
- FAQ clustering
- FAQ generation
- request validation
- LLM error handling

실제 Gemma response quality와 image recommendation 결과는 별도 수동 검증이 필요합니다.

## Project Structure

```text
ml/
├── ai_server/
├── ai_app/
│   ├── views.py
│   ├── urls.py
│   ├── inference.py
│   ├── retrieval.py
│   ├── clustering.py
│   ├── faq.py
│   ├── llm.py
│   ├── serializers.py
│   ├── application/
│   ├── infrastructure/
│   ├── management/commands/
│   └── inference_ipynb/
├── assets/
│   └── readme/
├── docs/
│   ├── qa_DB_tag_json.csv
│   ├── post_comments.json
│   ├── medical_metadata.json
│   ├── faq_drafts/
│   └── validation/
├── media/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── manage.py
└── requirements.txt
```

## Data & Validation Records

| Path | Purpose |
| --- | --- |
| `docs/qa_DB_tag_json.csv` | retrieval과 FAQ generation에 사용하는 과거 Q&A |
| `docs/post_comments.json` | 답변 context에 사용하는 과거 댓글 및 의료진 답변 |
| `docs/medical_metadata.json` | image recommendation용 description 및 tag |
| `docs/faq_drafts/` | FAQ draft와 review 상태 기록 |
| `docs/validation/retrieval-relevance-v1.json` | retrieval relevance label |
| `docs/validation/retrieval-benchmark-v1.json` | retrieval model benchmark |
| `docs/validation/faq-clustering-benchmark-v1.json` | FAQ clustering benchmark |
| `media/` | 원본 의료 참고 이미지 |

`docs/faq_clean_output.json`은 이전 FAQ generation 방식의 결과이며 현재 FAQ API에서는 사용하지 않습니다.
