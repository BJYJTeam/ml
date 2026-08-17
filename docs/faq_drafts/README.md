# FAQ Draft Runs

`generate_faq_drafts` 명령은 클러스터별 대표 과거 Q/A 쌍과 Gemma 출력을 하나의 버전 관리 JSON 실행 기록으로 저장합니다.

```bash
python manage.py generate_faq_drafts \
  --output docs/faq_drafts/faq-drafts-v1.json
```

새 초안은 항상 `pending_review` 상태로 저장됩니다. 의료진 또는 승인 권한자가 `approved`로 검토한 항목만 `GET /api/api/faqs/` 응답에 포함됩니다. `rejected` 항목과 파싱 오류가 난 항목은 공개하지 않습니다.

각 실행 기록에는 임베딩 모델, LLM 모델, HDBSCAN 설정, 대표 Q/A 수, 카테고리별 클러스터 진단, 생성 오류, 초안의 원본 게시글 ID를 함께 기록합니다.
