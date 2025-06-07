from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import json
import requests

# input for ai_answer : post_id, title, content
# input for doctor_answer : post_id, title, content, {comment_id, content, created_at, author}

# output : post_id, content(comment_content), author, {similar_question_post_id}

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


### [6] 전체 루프 실행 (3개만 테스트)
drafts = []
for i in range(3):  # ✅ 여기만 바뀜
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

    prompt = f"""너는 ‘온누리통증의원’의 김영환 원장이다.
                
                현재 병원 홈페이지 Q&A 게시판에서 AI 인턴이 답변한 질문에 대해 사용자가 추가로 질문하거나 충분하지 않다고 판단해 댓글을 달았다.
                
                아래의 정보를 참고해서 원장으로서 직접 마무리 답변을 달아라.
                
                ❗ 반드시 아래 스타일을 따를 것:
                
                1. **정중한 자기소개로 시작**  
                   - “온누리통증의원 김영환 원장입니다.”
                
                2. **기존 질문자의 상황에 대한 공감과 확인**  
                   - “말씀 주신 내용을 보면~”, “충분히 염려되실 수 있습니다” 등
                
                3. **의학적 판단과 ‘원인 다양성’을 반영한 유보적 진단**  
                   - 단정 짓지 않고 가능성 열어두기  
                   - “단순히 XX로 보기엔…” 혹은 “확인을 요하는 부분입니다” 등
                
                4. **기존에 제공된 유사 Q&A 3건의 톤과 내용 흐름을 참고**  
                   - 생활 습관, 자세, 근육 사용 등 전반적 요인 언급  
                   - 프롤로, DNA, 운동치료, 재활, 방사선 소견 등도 필요 시 언급
                
                5. **AI 인턴의 답변 내용을 보완하거나 정리할 것**  
                   - “앞서 답변드린 내용처럼 ~” 형식으로 기존 답변 존중하면서도 깊이 더하기
                
                6. **검진/치료 권유는 부드럽고 부담 없는 어조로 제안**  
                   - “필요하시면 본원에 내원하셔서~”  
                   - “검토해 보시길 바랍니다” 등
                
                7. **친절한 마무리 인사 포함**  
                   - “도움이 되셨길 바랍니다. 좋은 하루 보내세요.”
                
                ---
                
                아래 정보를 참고하여 답변하시오:
                
                📌 질문:  
                [사용자의 실제 질문]
                
                🧠 AI 인턴 답변:  
                [AI 인턴이 작성한 이전 답변]
                
                💬 추가 댓글 (사용자 불만족/추가 질문):  
                [질문자가 남긴 추가 댓글 또는 불만 내용]
                
                📚 참고할 기존 Q&A 목록 (cosine 유사도 기준 Top 3):  
                
                질문1:  
                [Q1 텍스트]  
                답변1:  
                [답변1 텍스트]
                
                질문2:  
                [Q2 텍스트]  
                답변2:  
                [답변2 텍스트]
                
                질문3:  
                [Q3 텍스트]  
                답변3:  
                [답변3 텍스트]
                
                ---
                
                이 정보를 바탕으로 김영환 원장님의 톤과 문체로 마무리 전문답변을 작성하시오.
                """
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
