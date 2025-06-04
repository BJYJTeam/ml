import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer

API_URL = "http://hanyang-datascience.duckdns.org:5005/run"
API_TOKEN = "z8y7x6w5v4.n1m2l3k4j5.Team4"

# 데이터 로드
df = pd.read_csv("qa_DB_tag_json.csv")
df = df[df['content'].notna()]  # content 없는 경우 제외
df = df[~df['content'].str.lower().str.contains('content not found')]  # 비밀글 제외
df["title"] = df["title"].fillna("")  # title NaN 방지

questions = df["title"].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

# API 요청 함수
def generate_gemma_answer(prompt: str) -> str:
    headers = {
        'Authorization': API_TOKEN,
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gemma3:12b',
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.ok:
        return response.json().get('response', '[응답 없음]')
    else:
        return f"[Error {response.status_code}] {response.text}"

def ai_answer(title, content, similar_questions):
    """
    일반 AI 답변 생성 함수
    - title: 질문 제목
    - content: 질문 본문
    - similar_questions: [{title, content, answer}, ...]
    """
    context = f"질문 제목: {title}\n질문 내용: {content}\n\n비슷한 질문들:\n"
    for i, sq in enumerate(similar_questions, 1):
        context += f"{i}. {sq.get('title', '')} - {sq.get('content', '')}\n"
    return f"AI 생성 답변 (입력 요약):\n{context}"


def doctor_answer(title, content, similar_questions):
    """
    의사 스타일 AI 답변 생성 함수
    - title: 질문 제목
    - content: 질문 본문
    - similar_questions: [{title, content, answer}, ...]
    """
    context = (
        "[척추측만증의원 전문의로서 답변합니다]\n\n"
        f"질문 제목: {title}\n질문 내용: {content}\n\n"
        "비슷한 질문들에 대한 이전 답변:\n"
    )
    for i, sq in enumerate(similar_questions, 1):
        context += f"{i}. {sq.get('title', '')} - {sq.get('content', '')}\n"
        if sq.get("answer"):
            context += f"→ 이전 답변: {sq['answer']}\n"

    return f"의사 스타일 답변 (입력 요약):\n{context}"
  
def extract_keywords_from_model(text):
  return text.split()[:3]

def get_faq_list():
    return [
        {"question": "무릎이 아파요", "answer": "병원에 방문하세요"},
        {"question": "척추측만증 수술 필요?", "answer": "정도에 따라 다릅니다"},
    ]
    
def find_similar_questions(title, content, top_n=5):
    query = f"{title}\n{content}"
    query_vec = model.encode([query])
    similarities = cosine_similarity(query_vec, question_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return df.iloc[top_indices][["Question", "Question Link", "Answer"]].to_dict(orient="records")


# test_api.py (예시 파일로 저장)
def test_api():
    result = generate_gemma_answer("척추측만증에 대해 설명해줘")
    print("API 응답:", result)

# test_api()



# --- Image Recommendation Based on Similarity ---
import json
import os

# Load precomputed image metadata
with open("medical_metadata.json", "r") as f:
    image_metadata = json.load(f)

image_descriptions = [item["description"] for item in image_metadata]
image_embeddings = model.encode(image_descriptions, convert_to_tensor=True).cpu()

def recommend_images_by_question(content, top_k=5):
    """
    질문 본문을 기반으로 가장 유사한 이미지 추천
    """
    query_embedding = model.encode([content], convert_to_tensor=True).cpu()
    similarities = cosine_similarity(query_embedding, image_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [image_metadata[i] for i in top_indices]