from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import json

### [1] 파일 불러오기
questions_df = pd.read_csv("qa_DB_tag_json.csv")
with open("cleaned_post_comments.json", "r", encoding="utf-8") as f:
    comments = json.load(f)

### [2] DOCTOR 답변을 post_id 기준으로 딕셔너리화
doctor_answers_by_post_id = {}
for c in comments:
    if c["author"].lower() == "doctor":
        doctor_answers_by_post_id.setdefault(c["post_id"], []).append(c["content"])

### [3] 전처리 및 임베딩 준비
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
questions_df = questions_df[questions_df['content'].notna()]
questions = questions_df['title'].fillna("") + " " + questions_df['content']
question_embeddings = model.encode(questions.tolist(), convert_to_tensor=True)

### [4] 유사 질문 찾기 함수
def find_similar_posts(query_index, top_n=3):
    query_embedding = question_embeddings[query_index]
    cosine_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_n + 1)
    similar_indices = [i.item() for i in top_results[1] if i.item() != query_index][:top_n]
    return similar_indices

### [5] context만 테스트 출력 (3개 질문에 대해)
for i in range(3):  # 상위 3개 질문만
    question = questions_df.iloc[i]
    title = question['title']
    content = question['content']
    post_id = question['id']
    
    similar_idxs = find_similar_posts(i)
    context = f"[질문 제목]: {title}\n[질문 내용]: {content}\n\n"
    context += "[유사 질문 및 전문의 답변 참고]\n"
    
    for idx in similar_idxs:
        sim_q = questions_df.iloc[idx]
        sim_post_id = sim_q['id']
        sim_answer = doctor_answers_by_post_id.get(sim_post_id, ['(답변 없음)'])[0]
        context += f"\nQ: {sim_q['title']}\nA: {sim_answer}"

    prompt = f"""당신은 측만증 전문 마취통증의학과 병원인 '온누리마취통증의학과의원'의 김영환 원장입니다. 환자의 상태를 배려하고, 신뢰를 줄 수 있도록 답변하세요.
아래 질문에 대해 전문의 스타일의 AI 답변 초안을 작성해주세요.

{context}"""

    print(f"\n\n=== [테스트 {i+1}] ===\n")
    print(prompt)
