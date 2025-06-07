#ai_app/inference.py

import json
from typing import Dict, List
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.cluster import HDBSCAN

# --- Constants and Model Initialization ---
API_URL = "http://hanyang-datascience.duckdns.org:5005/run"
API_TOKEN = "z8y7x6w5v4.n1m2l3k4j5.Team4"
MODEL_NAME = 'all-MiniLM-L6-v2'

model = SentenceTransformer(MODEL_NAME)


# --- Load QA Data and Preprocess ---
def load_qa_dataframe() -> pd.DataFrame:
    df = pd.read_csv("docs/qa_DB_tag_json.csv")
    df = df[df['content'].notna() & ~df['content'].str.lower().str.contains('content not found')]
    df["title"] = df["title"].fillna("")
    return df


questions_df = load_qa_dataframe()
question_embeddings = model.encode(questions_df["title"].tolist(), convert_to_tensor=True)


# --- Load Doctor Comments ---
def load_doctor_comments() -> Dict[int, List[str]]:
    with open("docs/post_comments.json", "r", encoding="utf-8") as f:
        comments = json.load(f)

    answers = {}
    for c in comments:
        if c["author"].lower() == "doctor":
            answers.setdefault(c["post_id"], []).append(c["content"])
    return answers


doctor_answers_by_post_id = load_doctor_comments()


# --- Gemma API Call ---
def generate_gemma_answer(prompt: str) -> str:
    headers = {
        'Authorization': API_TOKEN,
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gemma3:12b',
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json().get('response',
                               '[응답 없음]') if response.ok else f"[Error {response.status_code}] {response.text}"


# --- Find Similar Posts ---
def find_similar_posts(post_id: int, title: str, content: str, top_n: int = 3) -> List[int]:
    query = f"{title}\n{content}"
    query_embedding = model.encode([query], convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_indices = cosine_scores.topk(k=top_n)[1].cpu().tolist()
    return questions_df.iloc[top_indices]['id'].tolist()


# --- AI Intern Answer Generation ---
def generate_ai_intern_answer(post_id: int, title: str, content: str) -> Dict:
    similar_ids = find_similar_posts(post_id, title, content)
    context = f"[질문 제목]: {title}\n[질문 내용]: {content}\n\n[유사 질문 및 전문의 답변 참고]\n"
    for sid in similar_ids:
        row = questions_df[questions_df['id'] == sid].iloc[0]
        context += f"\nQ: {row['title']}\nA: {doctor_answers_by_post_id.get(sid, ['(답변 없음)'])[0]}"

    prompt = f"""당신은 \"온누리마취통증의학과의원\"에서 근무 중인 AI 인턴입니다.  
해당 병원은 마취통증의학과이며, 특히 측만증, 허리 통증, 자세 관련 치료에 전문성을 갖추고 있습니다.

아래는 새롭게 접수된 질문이며, 참고할 수 있도록 유사한 기존 질문 3건의 제목, 질문 내용, 답변이 함께 제공됩니다.  
이 정보를 바탕으로 환자나 보호자의 입장에서 이해하기 쉽고 따뜻한 답변을 작성해주세요.

[답변 작성 지침]
1. 답변은 **밝고 공손한 말투로 시작**하며,  
→ “안녕하세요. 온누리마취통증의학과의 AI 인턴입니다 :) 문의 주셔서 감사해요!”로 시작
2. 기존 유사 질문의 답변을 참고하여, **중복되지 않도록 자연스럽게 통합해서 새로운 질문에 맞게 답변 구성**
3. 진단이 필요한 부분은 **전문의의 진료가 필요하다는 점 안내**
4. **일상적 표현** 사용, **전문 용어는 지양**
5. 답변 마지막은 반드시 아래 문장으로 마무리
→ "감사합니다.\n예약 문의: 051-714-1831\n내원 전 예약 부탁드립니다."

{context}"""
    return {"post_id": post_id, "content": generate_gemma_answer(prompt).strip()}


# --- Doctor Draft Answer ---
def generate_doctor_draft(
        post_id: int,
        title: str,
        content: str,
        comments: List[Dict[str, str]]
) -> Dict:
    """
    여러 개의 댓글을 받아 원장님의 답변 초안을 생성합니다.

    Parameters:
        - post_id: 질문 글 ID
        - title: 질문 제목
        - content: 질문 본문
        - comments: [{comment_content: str, comment_author: str}, ...]

    Returns:
        - {'post_id': int, 'content': str}
    """
    similar_ids = find_similar_posts(post_id, title, content)

    context = f"질문:\n{title} {content}\n\n"
    for c in comments:
        author = c.get("comment_author", "사용자")
        text = c.get("comment_content", "")
        context += f"댓글 ({author}):\n{text}\n\n"

    context += "참고할 기존 Q&A 목록:\n"
    for sid in similar_ids:
        row = questions_df[questions_df['id'] == sid].iloc[0]
        context += f"\nQ: {row['title']}\nA: {doctor_answers_by_post_id.get(sid, ['(답변 없음)'])[0]}"

    prompt = f"""너는 ‘온누리통증의원’의 김영환 원장이다.

현재 병원 홈페이지 Q&A 게시판에서 AI 인턴이 답변한 질문에 대해 사용자가 추가로 질문하거나 충분하지 않다고 판단해 댓글을 달았다.

아래의 정보를 참고해서 원장으로서 직접 마무리 답변을 달아라.

! 반드시 아래 스타일을 따를 것:
1. “온누리통증의원 김영환 원장입니다.”로 시작
2. “말씀 주신 내용을 보면~” 등 공감 표현
3. 유보적 진단: “단순히 XX로 보기엔…”, “확인을 요하는 부분입니다”
4. 기존 Q&A 3건 톤 참고, 치료 예시 포함
5. “앞서 답변드린 내용처럼 ~” 형식으로 AI 답변 보완
6. 검진 권유는 부드럽게 제안: “내원하셔서~ 검토해 보시길 바랍니다”
7. “도움이 되셨길 바랍니다. 좋은 하루 보내세요.” 마무리

{context}"""

    return {"post_id": post_id, "content": generate_gemma_answer(prompt).strip()}



# --- FAQ Generation ---
def generate_faqs_from_db(num_questions: int = 3) -> List[Dict[str, str]]:
    df = questions_df.copy()
    df['category'] = df['tag'].str.extract(r'\[(.*?)\]')
    df['keyword_raw'] = df['tag'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
    df['keyword_list'] = df['keyword_raw'].str.split(r'[\s,]+')
    df['keyword_str'] = df['keyword_list'].apply(lambda x: ' '.join(x))

    target_clusters_per_category = {
        "증상 및 진단 문의": 4,
        "치료 및 시술 문의": 1,
        "예약, 진료, 비용 문의": 1,
        "생활관리 및 기타 문의": 1
    }

    faq_results = []
    for category_name, required_clusters in target_clusters_per_category.items():
        sub_df = df[df['category'] == category_name].copy().reset_index(drop=True)
        if len(sub_df) < 5:
            continue

        sub_df['full_text'] = sub_df['keyword_str'] + " " + sub_df['content']
        texts = sub_df['full_text'].tolist()
        embeddings = model.encode(texts)
        cosine_dist = (1 - cosine_similarity(embeddings)).astype(np.float64)

        min_cluster_size, min_samples = 4, 2
        valid_clusters = []
        for _ in range(10):
            clusterer = HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = clusterer.fit_predict(cosine_dist)
            sub_df['cluster'] = labels
            label_counts = Counter(labels)
            valid_clusters = [label for label in label_counts if label != -1]
            if len(valid_clusters) >= required_clusters:
                break
            min_cluster_size = max(2, min_cluster_size - 1)

        for cluster_id in sorted(valid_clusters, key=lambda x: label_counts[x], reverse=True)[:required_clusters]:
            cluster_df = sub_df[sub_df['cluster'] == cluster_id].reset_index(drop=True)
            if cluster_df.empty:
                continue

            cluster_embeddings = model.encode(cluster_df['full_text'].tolist())
            sim_matrix = cosine_similarity(cluster_embeddings)
            avg_similarities = (sim_matrix.sum(axis=1) - 1) / (len(sim_matrix) - 1)
            top_indices = np.argsort(avg_similarities)[-5:][::-1]

            questions_text = "\n".join([
                f"- 제목: {cluster_df.iloc[i]['title']}\n  질문: {cluster_df.iloc[i]['content']}" for i in top_indices
            ])

            representative_row = cluster_df.iloc[top_indices[0]]
            post_id = representative_row['id']
            doctor_reply = doctor_answers_by_post_id.get(post_id, ["(의사 답변 없음)"])[0]

            prompt = f"""[FAQ 생성 요청]
당신은 \"온누리마취통증의학과의원\"의 AI 인턴입니다.
다음은 \"{category_name}\" 카테고리의 대표 질문들이며, 실제 전문의 답변도 함께 제공됩니다.

[대표 질문들]
{questions_text}

[전문의 답변]
{doctor_reply}

위 질문들과 전문의 답변을 참고하여, FAQ 형식의 요약 질문/답변을 각각 1회씩 작성해주세요.

[출력 목적]
- 환자들이 자주 묻는 질문을 이해하고, 빠르게 답변을 확인할 수 있도록 합니다.

[출력 기준]
1. 질문은 반드시 **의문문**으로 한 문장으로 작성하세요.
2. 답변은 너무 길지 않게, 핵심만 담아 **간결하게** 작성하세요.
3. 중복되거나 불필요한 표현은 제거해주세요.

[출력 형식]
Q: 질문 내용
A: 답변 내용"""
            answer = generate_gemma_answer(prompt)
            lines = answer.split('\n')
            q_line = next((line for line in lines if "Q:" in line), "").strip()
            a_line = next((line for line in lines if "A:" in line), "").strip()

            faq_results.append({
                "content": q_line.replace("Q:", "").strip(),
                "answer": a_line.replace("A:", "").strip()
            })

            if len(faq_results) >= num_questions:
                return faq_results

    return faq_results


# --- Keyword Extraction ---
def extract_keywords(post_id: int, title: str, content: str) -> Dict:
    prompt = f"""다음은 마취통증의학과 병원에 올라온 환자 또는 보호자의 질문입니다.  
질문 내용을 바탕으로 다음 조건에 따라 의학적으로 중요한 핵심 키워드만 추출하세요.

[출력 목적]  
- 추출된 키워드는 의료진이 환자의 주요 증상, 가능 질환, 치료 방향 등을 빠르게 파악하고, 분류 및 검색 기능에 활용됩니다.

[추출 기준]  
1. 통증 부위, 증상의 성격 또는 유발 조건, 의심되는 질환명, 치료 방식만 추출하세요.  
2. 키워드는 Bi-gram(두 단어 조합)을 우선적으로 생성하고, 적절한 조합이 없을 경우에만 Uni-gram 사용 가능.  
3. 의미가 유사하거나 반복되는 표현은 하나로 병합하여 중복 없이 추출하세요.  
4. 행동이나 조건에 따른 통증이 나타날 경우, 이를 반영한 표현도 키워드로 추출하세요.  
5. 의학적으로 무의미한 단어는 제외하세요.  
6. 반드시 아래 카테고리 중 해당하는 하나를 키워드 목록의 **가장 앞에 포함**하세요.  
   - [증상 및 진단 문의]  
   - [치료 및 시술 문의]  
   - [예약, 진료, 비용 문의]  
   - [생활관리 및 기타 문의]

[출력 형식]  
- 쉼표로 구분된 키워드 목록만 출력하세요.
- 문장이나 설명은 포함하지 마세요.
- 카테고리는 항상 키워드 목록의 맨 앞에 위치해야 합니다.

—
"Title" : {title}
"Content" : {content}"""
    result = generate_gemma_answer(prompt)
    tags = [kw.strip() for kw in result.split(',') if kw.strip()]
    return {"post_id": post_id, "tag": tags}


# --- Image Recommendation ---
with open("docs/medical_metadata.json", "r") as f:
    image_metadata = json.load(f)

image_descriptions = [item["description"] for item in image_metadata]
image_embeddings = model.encode(image_descriptions, convert_to_tensor=True).cpu()


def recommend_images_by_question(content: str, top_k: int = 5) -> List[Dict]:
    query_embedding = model.encode([content], convert_to_tensor=True).cpu()
    similarities = cosine_similarity(query_embedding, image_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [image_metadata[i] for i in top_indices]