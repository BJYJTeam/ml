import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ai_app.faq import load_approved_faqs
from ai_app.infrastructure.embeddings import SentenceTransformerEncoder
from ai_app.llm import generate_gemma_answer
from ai_app.retrieval import QuestionDocument, SemanticRetriever

# --- Constants and Model Initialization ---
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
MODEL_NAME = 'all-MiniLM-L6-v2'
RETRIEVAL_MODEL_NAME = 'intfloat/multilingual-e5-small'
RETRIEVAL_MIN_SCORE = 0.9

# --- Load QA Data and Preprocess ---
@lru_cache(maxsize=1)
def load_qa_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DOCS_DIR / "qa_DB_tag_json.csv")
    df = df[df['content'].notna() & ~df['content'].str.lower().str.contains('content not found')]
    df["title"] = df["title"].fillna("")
    df["id"] = df["id"].astype(str)
    return df


@lru_cache(maxsize=1)
def get_retrieval_encoder() -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder(
        RETRIEVAL_MODEL_NAME,
        query_prefix="query: ",
        document_prefix="passage: ",
    )


@lru_cache(maxsize=1)
def get_question_retriever() -> SemanticRetriever:
    questions = load_qa_dataframe()
    return SemanticRetriever(
        [
            QuestionDocument(row.id, row.title, row.content)
            for row in questions.itertuples(index=False)
        ],
        get_retrieval_encoder(),
    )


# --- Load Doctor Comments ---
@lru_cache(maxsize=1)
def load_doctor_comments() -> Dict[str, List[str]]:
    with (DOCS_DIR / "post_comments.json").open(encoding="utf-8") as f:
        comments = json.load(f)

    answers = {}
    for c in comments:
        if c["author"].lower() == "doctor":
            answers.setdefault(str(c["post_id"]), []).append(c["content"])
    return answers


# --- Find Similar Posts ---
def find_similar_posts(
    title: str,
    content: str,
    top_n: int = 3,
    exclude_post_id: str | None = None,
) -> List[str]:
    """
    주어진 제목과 본문을 기반으로 가장 유사한 기존 질문 post_id 리스트를 반환합니다.

    Args:
        title (str): 질문 제목
        content (str): 질문 본문
        top_n (int): 반환할 유사 질문 수 (기본값 3)

    Returns:
        List[str]: 유사도가 높은 질문 ID 리스트
    """
    print("Start func of find_similar_posts")
    if not title and not content:
        return []

    print("Done find similar_posts")
    return [
        result.post_id
        for result in get_question_retriever().search(
            title,
            content,
            exclude_post_id=exclude_post_id,
            top_k=top_n,
            min_score=RETRIEVAL_MIN_SCORE,
        )
    ]


# --- AI Intern Answer Generation ---
def generate_ai_intern_answer(post_id: str, title: str, content: str) -> Dict:
    print("Start func of generate_ai_intern_answer")
    similar_ids = find_similar_posts(title, content, exclude_post_id=post_id)
    questions = load_qa_dataframe()
    doctor_answers = load_doctor_comments()
    context = f"[질문 제목]: {title}\n[질문 내용]: {content}\n\n[유사 질문 및 전문의 답변 참고]\n"
    for sid in similar_ids:
        row = questions[questions['id'] == sid].iloc[0]
        context += f"\nQ: {row['title']}\nA: {doctor_answers.get(sid, ['(답변 없음)'])[0]}"

    print("Run Prompt from generate_ai_intern_answer")
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
def generate_doctor_draft(post_id: str, title: str, content: str, comments: List[Dict[str, str]]) -> Dict:

    print("Start func of generate_doctor_draft")

    similar_ids = find_similar_posts(title, content, exclude_post_id=post_id)
    questions = load_qa_dataframe()
    doctor_answers = load_doctor_comments()

    # 댓글 분류
    ai_answer = ""
    user_comments = []

    for c in comments:
        author = c.get("comment_author", "").strip().lower()
        text = c.get("comment_content", "").strip()
        if author == "ai":
            ai_answer = text
        else:
            user_comments.append((author or "사용자", text))

    # 프롬프트 Context 구성
    context = f"[질문]\n{title}\n{content}\n\n"

    if ai_answer:
        context += f"[AI 인턴의 기존 답변]\n{ai_answer}\n\n"

    if user_comments:
        context += "[사용자 댓글]\n"
        for author, text in user_comments:
            context += f"({author}) {text}\n"

    # 유사 Q&A 추가
    context += "\n[참고할 기존 Q&A 목록]"
    for sid in similar_ids:
        row = questions[questions['id'] == sid].iloc[0]
        past_q = row['title']
        past_a = doctor_answers.get(sid, ['(답변 없음)'])[0]
        context += f"\n\nQ: {past_q}\nA: {past_a}"

    print("Run Prompt from generate_doctor_draft")

    # 최종 프롬프트
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


# --- FAQ Storage Facade ---
def generate_faqs_from_db() -> List[Dict[str, str]]:
    return load_approved_faqs()


# --- Keyword Extraction ---
def extract_keywords(post_id: str, title: str, content: str) -> Dict:
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
@lru_cache(maxsize=1)
def get_image_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_image_resources() -> tuple[List[Dict], object]:
    with (DOCS_DIR / "medical_metadata.json").open(encoding="utf-8") as f:
        image_metadata = json.load(f)
    image_descriptions = [item["description"] for item in image_metadata]
    image_embeddings = get_image_model().encode(
        image_descriptions, convert_to_tensor=True
    ).cpu()
    return image_metadata, image_embeddings


def recommend_images_by_question(content: str, top_k: int = 5) -> List[Dict]:
    image_metadata, image_embeddings = get_image_resources()
    query_embedding = get_image_model().encode([content], convert_to_tensor=True).cpu()
    similarities = cosine_similarity(query_embedding, image_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [image_metadata[i] for i in top_indices]
