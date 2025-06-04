import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from collections import Counter

# 1. 데이터 로드
df = pd.read_csv("./qa_DB_tag_json.csv")

# 2. 유효한 content만 필터링
df = df[df['content'].notna()]
df = df[~df['content'].str.lower().str.contains('content not found')].copy()

# 3. tag에서 category 및 키워드 분리
df['category'] = df['tag'].str.extract(r'\[(.*?)\]')
df['keyword_raw'] = df['tag'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
df['keyword_list'] = df['keyword_raw'].str.split(r'[,\s]+')
df['keyword_str'] = df['keyword_list'].apply(lambda x: ' '.join(x))

# 4. SBERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 5. 카테고리별 추출할 클러스터 수
target_clusters_per_category = {
    "증상 및 진단 문의": 4,
    "치료 및 시술 문의": 1,
    "예약, 진료, 비용 문의": 1,
    "생활관리 및 기타 문의": 1
}

faq_results = []
cluster_stats = {}

# 6. 카테고리별 처리
for category_name, required_clusters in target_clusters_per_category.items():
    sub_df = df[df['category'] == category_name].copy().reset_index(drop=True)
    if len(sub_df) < 5:
        continue

    # 🔁 keyword_str + content 병합해서 임베딩
    sub_df['full_text'] = sub_df['keyword_str'] + " " + sub_df['content']
    texts = sub_df['full_text'].tolist()
    embeddings = model.encode(texts)
    cosine_dist = (1 - cosine_similarity(embeddings)).astype(np.float64)

    min_cluster_size = 4
    min_samples = 2
    max_attempts = 10
    for attempt in range(max_attempts):
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(cosine_dist)
        sub_df['cluster'] = labels

        label_counts = Counter(labels)
        valid_clusters = [label for label in label_counts if label != -1]

        if len(valid_clusters) >= required_clusters:
            break
        min_cluster_size = max(2, min_cluster_size - 1)

    cluster_stats[category_name] = {
        "전체 군집 수": len(valid_clusters),
        "군집별 문서 수": {cid: label_counts[cid] for cid in valid_clusters},
        "실제 추출된 군집 수": min(required_clusters, len(valid_clusters))
    }

    sorted_cluster_ids = sorted(valid_clusters, key=lambda x: label_counts[x], reverse=True)[:required_clusters]

    for i, cluster_id in enumerate(sorted_cluster_ids, 1):
        cluster_df = sub_df[sub_df['cluster'] == cluster_id].copy().reset_index(drop=True)
        if cluster_df.empty:
            continue

        cluster_embeddings = model.encode(cluster_df['full_text'].tolist())  # 🔁 임베딩 기준도 변경
        sim_matrix = cosine_similarity(cluster_embeddings)
        avg_similarities = (sim_matrix.sum(axis=1) - 1) / (len(sim_matrix) - 1)
        top_indices = np.argsort(avg_similarities)[-5:][::-1]

        for idx in top_indices:
            row = cluster_df.iloc[idx]
            faq_results.append({
                "id": row['id'],
                "category": f"{category_name} - {i}",
                "title": row['title'],
                "content": row['content'],
                "keywords": row['keyword_list'],
                "similarity": round(avg_similarities[idx], 4)
            })

# 7. 결과 출력
for row in faq_results:
    print(f"[{row['category']}] {row['title']}")
    print("ID:", row['id'])
    print("키워드:", ', '.join(row['keywords']))
    print("내용:", row['content'][:200].replace('\n', ' ') + "...")
    print("유사도:", row['similarity'])
    print("------")

# 8. 군집 통계 출력
print("\n[군집 통계 요약]")
for category, stat in cluster_stats.items():
    print(f"\n▶ {category}")
    print(" - 전체 군집 수:", stat["전체 군집 수"])
    print(" - 군집별 문서 수:", stat["군집별 문서 수"])
    print(" - 실제 추출된 군집 수:", stat["실제 추출된 군집 수"])
