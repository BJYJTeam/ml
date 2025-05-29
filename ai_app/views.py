from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

import numpy as np

from .inference import (
    ai_answer,
    doctor_answer,
    extract_keywords_from_model,
    find_similar_questions,
    get_faq_list,
)

base_question_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=["title", "content"],
    properties={
        "postId": openapi.Schema(type=openapi.TYPE_STRING, description="postId"),
        "title": openapi.Schema(type=openapi.TYPE_STRING, description="질문 제목"),
        "content": openapi.Schema(type=openapi.TYPE_STRING, description="질문 내용"),
    },
    example={
        "postId": openapi.Schema(type=openapi.TYPE_STRING, description="postId"),
        "title": "무릎 통증이 있어요",
        "content": "계단 오르내릴 때 무릎이 아픕니다.",
    }
)

answer_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=["title", "content", "similar_questions"],
    properties={
        "postId": openapi.Schema(type=openapi.TYPE_STRING, description="postId"),
        "title": openapi.Schema(type=openapi.TYPE_STRING, description="질문 제목"),
        "content": openapi.Schema(type=openapi.TYPE_STRING, description="질문 내용"),
        "similar_questions": openapi.Schema(
            type=openapi.TYPE_ARRAY,
            items=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "title": openapi.Schema(type=openapi.TYPE_STRING),
                    "content": openapi.Schema(type=openapi.TYPE_STRING),
                    "answer": openapi.Schema(type=openapi.TYPE_STRING)
                }
            ),
            description="비슷한 질문 리스트"
        )
    },
    example={
        "postId": "1",
        "title": "무릎이 아파요",
        "content": "계단을 오르내릴 때 무릎에 통증이 있어요.",
        "similar_questions": [
            {
                "title": "계단 내려갈 때 무릎 통증",
                "content": "무릎이 아프고 붓습니다.",
                "answer": "계단 내려갈 때 통증이 있다면 연골 손상 가능성이 있으니 병원 방문이 필요합니다."
            },
            {
                "title": "무릎이 붓고 아파요",
                "content": "오랫동안 서 있거나 걸으면 아파요.",
                "answer": "휴식과 냉찜질을 해보고, 통증이 지속되면 정형외과 진료를 권장합니다."
            }
        ]
    }
)

# answer 
# param postId, title, content
# return postId, AI 답변 (content)
@swagger_auto_schema(method='post', request_body=answer_schema)
@api_view(['POST'])
def answer(request):
    title = request.data.get("title", "")
    content = request.data.get("content", "")
    similar_questions = request.data.get("similar_questions", [])
    return Response({"answer": ai_answer(title, content, similar_questions)})

# doctor answer
# param postId, title, content, tag, comment
# return postId, 의사용 AI 답변 (content)
@swagger_auto_schema(method='post', request_body=answer_schema)
@api_view(['POST'])
def doctor_answer_view(request):  
    title = request.data.get("title", "")
    content = request.data.get("content", "")
    similar_questions = request.data.get("similar_questions", [])
    return Response({"answer": doctor_answer(title, content, similar_questions)})

# keyword
# param postId, title, content
# return postId, content (tag list형식)
@swagger_auto_schema(method='post', request_body=base_question_schema)
@api_view(['POST'])
def extract_keywords(request):
    text = request.data.get('text', '')
    return Response({"keywords": extract_keywords_from_model(text)})

# faq
# param postId, title, content, tag
# return list로 (title, content, category) 
@api_view(['GET'])
def faqs(request):
    return Response({"faqs": get_faq_list()})


@swagger_auto_schema(
    method='post',
    request_body=base_question_schema,
    responses={
        200: openapi.Response(
            description="유사한 질문 목록",
            examples={
                "application/json": {
                    "similar_questions": [
                        {
                            "Question": "무릎이 아플 때 어떻게 하나요?",
                            "Question Link": "/questions/1",
                            "Answer": "휴식과 병원 방문을 추천합니다."
                        }
                    ]
                }
            }
        )
    }
)
@api_view(['POST'])
def similar_questions(request):
    title = request.data.get("title", "")
    content = request.data.get("content", "")
    results = find_similar_questions(title, content)
    if "error" in results:
        return Response(results, status=500)
    return Response({"similar_questions": results})