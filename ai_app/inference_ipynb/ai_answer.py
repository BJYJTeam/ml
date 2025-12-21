from sentence_transformers import SentenceTransformer, util
import os
import torch
import pandas as pd
import json
import requests

# input for ai_answer : post_id, title, content
# input for doctor_answer : post_id, title, content, {comment_id, content, created_at, author}

# output : post_id, content(comment_content), author, {similar_question_post_id}

API_URL = os.getenv("API_URL", "")
API_TOKEN = os.getenv("GEMMA_API_TOKEN", "")

### [1] 파일 불러오기
questions_df = pd.read_csv("qa_DB_tag_json.csv")
with open("post_comments.json", "r", encoding="utf-8") as f:
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
    top_results = torch.topk(cosine_scores, k=top_n + 1)  # 자기 자신 포함됨
    similar_indices = [i.item() for i in top_results[1] if i.item() != query_index][:top_n]
    return similar_indices

### [5] GEMMA API 호출 함수
def generate_gemma_answer(prompt: str) -> str:
    if not API_URL or not API_TOKEN:
        return "[Error: API_URL or GEMMA_API_TOKEN not set]"
    headers = {
        'Authorization': API_TOKEN,
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gemma3:12b',
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post(API_URL, headers=headers, json=data)
    if response.ok:
        return response.json().get('response', '[응답 없음]')
    return f"[Error {response.status_code}] {response.text}"

### [6] 전체 루프 실행 (3개만 테스트)
drafts = []
for i in range(3):
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
        context += f"\nQ: {sim_q['title']}\nA: {doctor_answers_by_post_id.get(sim_post_id, ['(답변 없음)'])[0]}"

    prompt = f"""당신은 "온누리마취통증의학과의원"에서 근무 중인 AI 인턴입니다.  
            해당 병원은 마취통증의학과이며, 특히 측만증, 허리 통증, 자세 관련 치료에 전문성을 갖추고 있습니다.

            아래는 새롭게 접수된 질문이며, 참고할 수 있도록 유사한 기존 질문 3건의 제목, 질문 내용, 답변이 함께 제공됩니다.  
            이 정보를 바탕으로 환자나 보호자의 입장에서 이해하기 쉽고 따뜻한 답변을 작성해주세요.

            [답변 작성 지침]

            1. 답변은 **밝고 공손한 말투로 시작**하며,  
            첫 문장은 반드시 다음과 같은 인사로 시작하세요:  
            → “안녕하세요. 온누리마취통증의학과의 AI 인턴입니다 :) 문의 주셔서 감사해요!”

            2. 기존 유사 질문의 답변을 참고하여,  
            **중복되지 않도록 자연스럽게 통합해서 새로운 질문에 맞게 답변을 구성**하세요.

            3. 진단이 필요한 부분이나 정보가 부족한 부분은,  
            **전문의의 정확한 진료가 필요하다는 점을 안내**하세요.  

            4. 너무 전문적인 용어보다는,  
            **일반인도 이해하기 쉬운 일상적인 표현을 사용**해주세요.

            5. 답변 마지막은 반드시 아래 문장으로 마무리하세요:  
            → "감사합니다.\n\n"
              "예약 문의: 051-714-1831\n"
              "내원 전 예약 부탁드립니다."

            [새로운 질문]  
            신규_질문_본문

            [유사 질문 1]  
            제목: 유사_질문_1_제목  
            질문: 유사_질문_1_본문  
            답변: 유사_질문_1_답변  

            [유사 질문 2]  
            제목: 유사_질문_2_제목  
            질문: 유사_질문_2_본문  
            답변: 유사_질문_2_답변  

            [유사 질문 3]  
            제목: 유사_질문_3_제목  
            질문: 유사_질문_3_본문  
            답변: 유사_질문_3_답변  

            [출력 형식]

            - 답변은 따뜻하고 부드러운 어투로 작성하세요.  
            - 첫 문장은 “안녕하세요. 온누리마취통증의학과의 AI 인턴입니다 :) 문의 주셔서 감사해요!”로 시작하세요.  

             \n\n{context}"""
    ai_draft = generate_gemma_answer(prompt)

    drafts.append({
        "post_id": post_id,
        "title": title,
        "content": content,
        "ai_draft": ai_draft
    })

    print(f"[{i + 1}/3] 완료: {title}")

### [7] 결과 저장
pd.DataFrame(drafts).to_csv("ai_draft_answers.csv", index=False, encoding='utf-8-sig')
