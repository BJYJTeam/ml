import pandas as pd
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from collections import Counter

# input : {title, content, {author가 "doctor"인 댓글들}}
# output : FAQ_id, question, answer

# ------------------------------
# 1. 데이터 로드 및 전처리
# ------------------------------
df = pd.read_csv("docs/qa_DB_tag_json.csv")

df = df[df['content'].notna()]
df = df[~df['content'].str.lower().str.contains('content not found')].copy()

df['category'] = df['tag'].str.extract(r'\[(.*?)\]')
df['keyword_raw'] = df['tag'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
df['keyword_list'] = df['keyword_raw'].str.split(r'[,\s]+')
df['keyword_str'] = df['keyword_list'].apply(lambda x: ' '.join(x))

# ------------------------------
# 2. SBERT 모델 로드
# ------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# 3. 카테고리별 클러스터 수 지정
# ------------------------------
target_clusters_per_category = {
    "증상 및 진단 문의": 4,
    "치료 및 시술 문의": 1,
    "예약, 진료, 비용 문의": 1,
    "생활관리 및 기타 문의": 1
}

faq_results = []
cluster_stats = {}

# ------------------------------
# 4. 카테고리별 군집화 및 대표 질문 추출
# ------------------------------
for category_name, required_clusters in target_clusters_per_category.items():
    sub_df = df[df['category'] == category_name].copy().reset_index(drop=True)
    if len(sub_df) < 5:
        continue

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

        cluster_embeddings = model.encode(cluster_df['full_text'].tolist())
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

# ------------------------------
# 5. GEMMA 응답 생성 함수 정의
# ------------------------------
def generate_gemma_answer(prompt: str) -> str:
    headers = {
        'Authorization': 'z8y7x6w5v4.n1m2l3k4j5.Team4',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gemma3:12b',
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post("http://hanyang-datascience.duckdns.org:5005/run", headers=headers, json=data)
    if response.ok:
        return response.json().get('response', '[응답 없음]')
    return f"[Error {response.status_code}] {response.text}"

# ------------------------------
# 6. 각 군집에서 대표 질문 1개씩만 선별
# ------------------------------
faq_per_cluster = {}
for row in faq_results:
    key = row['category']
    if key not in faq_per_cluster:
        faq_per_cluster[key] = row  # 첫 항목(=유사도 높은 대표 질문)

# ------------------------------
# 7. GEMMA에 프롬프트 전달하여 FAQ 응답 생성 (전문의 답변 포함)
# ------------------------------
# 📌 댓글 JSON 불러오기
with open("post_comments.json", "r", encoding="utf-8") as f:
    comments = json.load(f)

# post_id → doctor 답변 딕셔너리 생성
doctor_answer_by_post_id = {}
for c in comments:
    if c["author"].lower() == "doctor":
        doctor_answer_by_post_id.setdefault(c["post_id"], []).append(c["content"])

generated_faqs = []

for category, faq in faq_per_cluster.items():
    post_id = faq["id"]
    doctor_reply = doctor_answer_by_post_id.get(post_id, ["(의사 답변 없음)"])[0]

    prompt = f"""[FAQ 생성 요청]
당신은 "온누리마취통증의학과의원"의 AI 인턴입니다.
다음은 "{category}" 카테고리의 대표 질문이며, 실제 전문의 답변도 함께 제공됩니다.

제목: {faq['title']}
질문 내용: {faq['content']}

[전문의 답변]
{doctor_reply}

위 질문과 전문의 답변을 참고하여, FAQ 형식의 요약 답변을 작성해주세요.
-질문은 의문문을 사용해 한 문장으로 나타해주세요
-질문과 답변은 각각 1회만 출력해주세요.  
-중복 없이 간결하게 작성해주세요.
-질문 형식은 "Q: 질문내용", 답변형식은 "A: 답변내용" 으로 나타내주세요.
"""
    answer = generate_gemma_answer(prompt)
    generated_faqs.append({
        "category": category,
        "question": faq['title'],
        "content": faq['content'],
        "doctor_answer": doctor_reply,
        "gemma_answer": answer
    })


# ------------------------------
# 8. 결과 출력
# ------------------------------
for faq in generated_faqs:
    #print(f"[{faq['category']}] {faq['question']}")
    #print(f"질문 내용: {faq['content'][:150]}...")
    print(f"▶ GEMMA 응답:\n{faq['gemma_answer']}")
    print("------")


# ------------------------------
# 9. 저장 형식 변환 및 JSON 저장
# ------------------------------
cleaned_faqs = []

for faq in generated_faqs:
    lines = faq['gemma_answer'].split('\n')
    
    # 질문과 답변 줄 추출
    q_line = next((line for line in lines if "Q:" in line), "").strip()
    a_line = next((line for line in lines if "A:" in line), "").strip()

    
    q_text = (
        q_line.replace("Q: ", "")
    )

    a_text = (
        a_line.replace("A: ", "")
    )
    cleaned_faqs.append({
        "id": "",
        "question": q_text,
        "answer": a_text
    })

# 저장
with open("faq_clean_output.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_faqs, f, ensure_ascii=False, indent=2)

print("저장 완료: faq_clean_output.json")

