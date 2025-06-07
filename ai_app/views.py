from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .inference import (
    generate_faqs_from_db,
    generate_doctor_draft,
    extract_keywords,
    generate_ai_intern_answer,
    find_similar_posts,
)


@swagger_auto_schema(
    method='get',
    operation_summary="유사 질문 ID 조회",
    manual_parameters=[
        openapi.Parameter('post_id', openapi.IN_QUERY, type=openapi.TYPE_STRING),
        openapi.Parameter('title', openapi.IN_QUERY, type=openapi.TYPE_STRING),
        openapi.Parameter('content', openapi.IN_QUERY, type=openapi.TYPE_STRING),
    ],
    responses={
        200: openapi.Response(
            description='유사 질문 ID 리스트',
            schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'post_id': openapi.Schema(type=openapi.TYPE_STRING),
                    'similar_post_ids': openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_STRING)
                    )
                }
            ),
            examples={
                "application/json": {
                    "post_id": "c47b9e89-8ac5-4ff4-aedb-0bd4d0ccadf4",
                    "similar_post_ids": [
                        "1e2d7b90-99e1-4e60-b754-5a5a54f5073a",
                        "3cdd2104-6fd6-4fd8-bf8e-c12ed1c939ed"
                    ]
                }
            }
        )
    }
)
@api_view(['GET'])
def similar_posts_api(request):
    post_id = request.GET.get("post_id")
    title = request.GET.get("title", "")
    content = request.GET.get("content", "")

    similar_ids = find_similar_posts(title, content)
    return Response({
        "post_id": post_id,
        "similar_post_ids": similar_ids
    })


@swagger_auto_schema(
    method='post',
    operation_summary="AI 인턴 답변 생성",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['post_id', 'title', 'content'],
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'title': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        }
    ),
    responses={200: openapi.Response('AI 답변', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        }
    ))}
)
@api_view(['POST'])
def ai_answer_view(request):
    data = request.data
    result = generate_ai_intern_answer(data['post_id'], data['title'], data['content'])
    return Response(result)


@swagger_auto_schema(
    method='post',
    operation_summary="의사 초안 답변 생성",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['post_id', 'title', 'content', 'comments'],
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'title': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING),
            'comments': openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    required=["comment_content", "comment_author"],
                    properties={
                        "comment_content": openapi.Schema(type=openapi.TYPE_STRING),
                        "comment_author": openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    ),
    responses={200: openapi.Response('의사 초안 답변', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        }
    ))}
)
@api_view(['POST'])
def doctor_draft_view(request):
    data = request.data
    post_id = data['post_id']
    title = data['title']
    content = data['content']
    comments = data['comments']
    result = generate_doctor_draft(post_id, title, content, comments)
    return Response(result)


@swagger_auto_schema(
    method='post',
    operation_summary="키워드 추출",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['post_id', 'title', 'content'],
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'title': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        }
    ),
    responses={200: openapi.Response('키워드 추출 결과', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'tag': openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Schema(type=openapi.TYPE_STRING)
            )
        }
    ))}
)
@api_view(['POST'])
def keyword_extraction_view(request):
    data = request.data
    result = extract_keywords(data['post_id'], data['title'], data['content'])
    return Response(result)


@swagger_auto_schema(
    method='get',
    operation_summary="FAQ 추출",
    responses={200: openapi.Response('FAQ 리스트', schema=openapi.Schema(
        type=openapi.TYPE_ARRAY,
        items=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'content': openapi.Schema(type=openapi.TYPE_STRING),
                'answer': openapi.Schema(type=openapi.TYPE_STRING)
            }
        )
    ))}
)
@api_view(['GET'])
def faq_list_view(request):
    faqs = generate_faqs_from_db()
    return Response(faqs)


@swagger_auto_schema(
    method='post',
    operation_summary="추천 이미지 제공",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=["content"],
        properties={
            "content": openapi.Schema(type=openapi.TYPE_STRING, description="질문 내용"),
        },
        example={
            "content": "청소년의 척추측만증 보조기 치료에 대해 알고 싶어요."
        }
    ),
    responses={
        200: openapi.Response(
            description="추천 이미지 리스트",
            examples={
                "application/json": {
                    "results": [
                        {
                            "id": "scoliosis-xray-1",
                            "title": "청소년 척추측만증 X-ray",
                            "description": "15세 환자의 25도 콥스 각도 척추측만증 X-ray",
                            "url": "https://cdn.com/image1.jpg",
                            "score": 0.91
                        }
                    ]
                }
            }
        )
    }
)
@api_view(['POST'])
def recommend_images(request):
    content = request.data.get("content", "")
    if not content:
        return Response({"error": "content is required"}, status=400)

    from .inference import recommend_images_by_question
    results = recommend_images_by_question(content)
    return Response({"results": results})
