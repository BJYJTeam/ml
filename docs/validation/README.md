# Retrieval Validation Set

This directory contains reviewed relevance judgments used to evaluate semantic retrieval models and thresholds. It must not contain direct patient identifiers or unredacted consultation text unless the repository's data-governance policy explicitly permits it.

It also stores versioned benchmark result records. Keep the validation-set version and the result-record version together so a model-selection decision can be reproduced and audited.

## Required Format

Create a versioned JSON file such as `retrieval-relevance-v1.json` from `retrieval-relevance.example.json`.

Each record must contain:

- `query_id`: stable identifier for the evaluation question.
- Either `source_post_id`, or redacted `title` and `content`.
- `relevant_post_ids`: historical Q&A IDs judged relevant by a reviewer.
- `reviewer`: reviewer identifier or role.
- `reviewed_at`: ISO 8601 review date.

`source_post_id` is preferred when the evaluation query is sampled from the repository's historical Q&A corpus. The benchmark resolves its title and content from `docs/qa_DB_tag_json.csv` and excludes that same post from the search result.

## Review Rules

- Include 30-50 questions across all four existing consultation categories.
- Mark a historical Q&A as relevant only when it would be appropriate context for an answer draft.
- Record disagreements in the pull request or experiment report before modifying relevance judgments.
- Increment the dataset version whenever judgments, source questions, or review rules change.

## Benchmark Result Records

Run the benchmark and save its output under this directory:

```bash
python manage.py benchmark_retrieval_models \
  --output docs/validation/retrieval-benchmark-v1.json
```

Each record must identify the validation data, candidate models, thresholds, retrieval metrics, selection criterion, and selected model. `retrieval-benchmark-v1.json` is the record used for the current production similar-question retriever.

## FAQ Clustering Benchmark Records

Run the clustering experiment and save its complete output under this directory:

```bash
python manage.py benchmark_faq_clustering \
  --output docs/validation/faq-clustering-benchmark-v1.json
```

The record must retain every compared configuration, category-level diagnostics, cluster sizes, representative-question samples, the automatic score, and the documented production-selection decision. `faq-clustering-benchmark-v1.json` is the record for the current FAQ candidate-clustering configuration.
