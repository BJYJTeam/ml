from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .faq import load_approved_faqs
from .inference import (
    generate_doctor_draft,
    extract_keywords,
    generate_ai_intern_answer,
    find_similar_posts,
)
from .llm import LLMError
from .serializers import (
    DoctorDraftSerializer,
    ImageRecommendationSerializer,
    MedicalQuestionSerializer,
)


@swagger_auto_schema(
    method='post',
    operation_summary="유사 질문 ID 조회",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['post_id', 'title', 'content'],
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'title': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        },
        example={
            "post_id": "0115181c415c4cdcafa51db1bfc2ea99",
            "title": "협착증도 치료 가능한가요",
            "content": "안녕하세요\n허리 협착증도 치료 가능한가요\n저희 어머니가 협착증으로 오래도록 고생하고 계신데 치료를 어떻게 해야 하는지 해서\n저희 어머님이 가신 병원도 수술여부를 저희 보고 결정하라고 하시는데..\n연세도 있으시고 주위에 수술 하신분들도 여전히 아프시다고 하셔서 수술을 해도 될런지 궁금합니다.\n혹 치료를 진행 한다면 어떤 치료를 하는지도 궁금 하구요\n답변 부탁 드립니다."
        }
    ),
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
                    "post_id": "0115181c415c4cdcafa51db1bfc2ea99",
                    "similar_post_ids": [
                        "792b9f577be746678fc4fb20b5a28c44",
                        "46f9c1b5a6cb459b8f5b9aea6aaaf406",
                        "4a41d08180664985ab462409fe34fe80"
                    ]
                }
            }
        )
    }
)
@api_view(['POST'])
def similar_posts_view(request):
    """
    주어진 질문 제목과 본문을 기반으로 유사한 기존 질문 ID 리스트를 반환합니다.
    """
    serializer = MedicalQuestionSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)
    data = serializer.validated_data
    post_id = data["post_id"]
    title = data["title"]
    content = data["content"]

    similar_ids = find_similar_posts(title, content, exclude_post_id=post_id)
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
        },
        example={
            'post_id': "0115181c415c4cdcafa51db1bfc2ea99",
            'title': "협착증도 치료 가능한가요",
            'content': "안녕하세요\n허리 협착증도 치료 가능한가요\n저희 어머니가 협착증으로 오래도록 고생하고 계신데 치료를 어떻게 해야 하는지 해서\n저희 어머님이 가신 병원도 수술여부를 저희 보고 결정하라고 하시는데..\n연세도 있으시고 주위에 수술 하신분들도 여전히 아프시다고 하셔서 수술을 해도 될런지 궁금합니다.\n혹 치료를 진행 한다면 어떤 치료를 하는지도 궁금 하구요\n답변 부탁 드립니다."
        }
    ),
    responses={200: openapi.Response('AI 답변', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        },
        examples={
            "application/json": {
                'post_id': "0115181c415c4cdcafa51db1bfc2ea99",
                'content': "안녕하세요. 온누리마취통증의학과 AI 인턴입니다 :) 문의 주셔서 감사해요!\n\n어머니의 허리 협착증으로 걱정이 많으신 것 같습니다. 협착증은 척추 채널이 좁아져 신경을 압박하고 통증을 유발하는 질환으로, 온누리마취통증의학과에서는 다양한 방법으로 협착증으로 인한 통증을 관리하고 있습니다.\n\n수술 여부는 어머니의 상태, 통증 정도, 기저 질환 등을 종합적으로 고려하여 결정해야 합니다. 환자분께서도 수술 후유증이나 주변 분들의 경험으로 인해 걱정이 많으신 것 같아, 수술 외에 다른 치료 방법들을 먼저 고려해 볼 수 있습니다. \n\n온누리에서는 신경 차단술, 프롤로 치료, 신경 재생 치료 등 비수술적 치료를 통해 통증을 완화하고 기능을 개선하는 데 집중할 수 있습니다. 물론, 정확한 진단과 함께 환자분께 가장 적합한 치료 계획을 세우기 위해서는 전문의의 진료가 필수적입니다.\n\n현재 어머니의 상태를 정확히 파악하고, 최적의 치료 방법을 결정하기 위해 꼭 진료를 받아보시길 권해드립니다.\n\n감사합니다.\n예약 문의: 051-714-1831\n내원 전 예약 부탁드립니다."
            }
        }
    ))}
)
@api_view(['POST'])
def ai_answer_view(request):
    serializer = MedicalQuestionSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)
    try:
        result = generate_ai_intern_answer(**serializer.validated_data)
        return Response(result)
    except LLMError:
        return Response({"error": "AI generation is temporarily unavailable."}, status=503)


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
                    required=["comment_author", "comment_content"],
                    properties={
                        "comment_author": openapi.Schema(type=openapi.TYPE_STRING),
                        "comment_content": openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        },
        example={
            'post_id': "0115181c415c4cdcafa51db1bfc2ea99",
            'title': "협착증도 치료 가능한가요",
            'content': "안녕하세요\n허리 협착증도 치료 가능한가요\n저희 어머니가 협착증으로 오래도록 고생하고 계신데 치료를 어떻게 해야 하는지 해서\n저희 어머님이 가신 병원도 수술여부를 저희 보고 결정하라고 하시는데..\n연세도 있으시고 주위에 수술 하신분들도 여전히 아프시다고 하셔서 수술을 해도 될런지 궁금합니다.\n혹 치료를 진행 한다면 어떤 치료를 하는지도 궁금 하구요\n답변 부탁 드립니다.",
            "comments": [
                {
                    "comment_author": "AI",
                    "comment_content": "안녕하세요. 온누리마취통증의학과 AI 인턴입니다 :) 문의 주셔서 감사해요!\n\n어머니의 허리 협착증으로 걱정이 많으신 것 같습니다. 협착증은 척추 채널이 좁아져 신경을 압박하고 통증을 유발하는 질환으로, 온누리마취통증의학과에서는 다양한 방법으로 협착증으로 인한 통증을 관리하고 있습니다.\n\n수술 여부는 어머니의 상태, 통증 정도, 기저 질환 등을 종합적으로 고려하여 결정해야 합니다. 환자분께서도 수술 후유증이나 주변 분들의 경험으로 인해 걱정이 많으신 것 같아, 수술 외에 다른 치료 방법들을 먼저 고려해 볼 수 있습니다. \n\n온누리에서는 신경 차단술, 프롤로 치료, 신경 재생 치료 등 비수술적 치료를 통해 통증을 완화하고 기능을 개선하는 데 집중할 수 있습니다. 물론, 정확한 진단과 함께 환자분께 가장 적합한 치료 계획을 세우기 위해서는 전문의의 진료가 필수적입니다.\n\n현재 어머니의 상태를 정확히 파악하고, 최적의 치료 방법을 결정하기 위해 꼭 진료를 받아보시길 권해드립니다.\n\n감사합니다.\n예약 문의: 051-714-1831\n내원 전 예약 부탁드립니다."

                },
                {
                    "comment_author": "USER",
                    "comment_content": "운동 능력 많이 떨어지는 분들에게 추천해주시는 비수술적 치료는 어떤게 있나요?"
                }
            ]
        }
    ),
    responses={200: openapi.Response('의사 초안 답변', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'post_id': openapi.Schema(type=openapi.TYPE_STRING),
            'content': openapi.Schema(type=openapi.TYPE_STRING)
        },
        examples={
            "application/json": {
                'post_id': "0115181c415c4cdcafa51db1bfc2ea99",
                'content': "온누리통증의원 김영환 원장입니다.\n\n말씀 주신 내용을 보면 어머님께서 허리 협착증으로 오랫동안 고생하시고, 수술에 대한 고민이 많으신 상황이네요. 사용자님께서 어머님의 운동성이 많이 떨어지시는 경우 비수술적 치료에 대한 궁금증을 주셨는데, 꼼꼼히 설명드리겠습니다.\n\n협착증은 척추 채널이 좁아져 신경을 압박하는 질환으로, 운동성이 떨어지는 경우 더욱 불편함을 느끼실 수 있습니다. 단순히 운동 부족으로 인한 불편함과는 구별해야 하며, 척추의 변형이나 퇴행성 변화가 원인일 수 있습니다.\n\n온누리통증의원에서는 운동성이 제한된 협착증 환자분들을 위해 다음과 같은 비수술적 치료를 고려합니다.\n\n*   **신경 차단술:** 좁아진 신경 주변에 약물을 주입하여 염증을 줄이고 통증을 완화합니다.\n*   **프롤로 치료:** 손상된 인대 조직을 강화하여 척추의 안정성을 높이고 통증을 줄입니다.\n*   **신경 재생 치료:** 손상된 신경 조직의 회복을 촉진하여 신경 기능을 개선합니다.\n*   **도수 치료:** 숙련된 치료사가 직접 손으로 척추와 주변 조직을 교정하여 운동 범위를 늘리고 통증을 완화합니다.\n*   **물리 치료:** 온열 치료, 전기 자극 치료, 초음파 치료 등을 통해 근육을 이완시키고 혈액 순환을 개선하여 통증을 줄입니다.\n\n단순히 협착증으로 보기에는 어머님의 상태가 복합적일 수 있으며, 정확한 진단을 위해서는 MRI 등의 정밀 검사가 필요합니다. 수술은 최후의 방법이며, 비수술적 치료로 효과를 보지 못할 경우 고려할 수 있습니다.\n\n앞서 답변드린 내용처럼, 어머님의 상태를 정확히 파악하고 최적의 치료 계획을 세우기 위해 내원하셔서 정밀 검사와 전문의의 진료를 받아 보시길 권해드립니다.\n\n도움이 되셨길 바랍니다. 좋은 하루 보내세요."
            }
        }
    ))}
)
@api_view(['POST'])
def doctor_draft_view(request):
    serializer = DoctorDraftSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)
    try:
        result = generate_doctor_draft(**serializer.validated_data)
        return Response(result)
    except LLMError:
        return Response({"error": "AI generation is temporarily unavailable."}, status=503)


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
        },
        example={
            'post_id': "0115181c415c4cdcafa51db1bfc2ea99",
            'title': "협착증도 치료 가능한가요",
            'content': "안녕하세요\n허리 협착증도 치료 가능한가요\n저희 어머니가 협착증으로 오래도록 고생하고 계신데 치료를 어떻게 해야 하는지 해서\n저희 어머님이 가신 병원도 수술여부를 저희 보고 결정하라고 하시는데..\n연세도 있으시고 주위에 수술 하신분들도 여전히 아프시다고 하셔서 수술을 해도 될런지 궁금합니다.\n혹 치료를 진행 한다면 어떤 치료를 하는지도 궁금 하구요\n답변 부탁 드립니다."
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
        },
        examples={
            "application/json": {
                'post_id': "0115181c415c4cdcafa51db1bfc2ea99",
                'tag': [
                    "[증상 및 진단 문의]",
                    "협착증",
                    "허리 협착증",
                    "수술",
                    "치료",
                    "통증"
                ]
            }
        }
    ))}
)
@api_view(['POST'])
def keyword_extraction_view(request):
    serializer = MedicalQuestionSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)
    try:
        result = extract_keywords(**serializer.validated_data)
        return Response(result)
    except LLMError:
        return Response({"error": "AI generation is temporarily unavailable."}, status=503)


@swagger_auto_schema(
    method='get',
    operation_summary="카테고리 기반 FAQ 자동 생성",
    responses={
        200: openapi.Response(
            description="FAQ 리스트",
            schema=openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'content': openapi.Schema(type=openapi.TYPE_STRING, description="FAQ 질문"),
                        'answer': openapi.Schema(type=openapi.TYPE_STRING, description="FAQ 답변"),
                    },
                    example= [
                        {
                            "content": "** 성인 측만증은 치료가 가능한가요?**",
                            "answer": "성인 측만증의 경우 환자분의 노력과 전문적인 치료가 필요합니다. 저희 병원에서는 운동 지도와 함께 통증 개선 및 예방 치료를 병행하고 있습니다. 자세한 상담은 내원하여 문의해주세요."
                        },
                        {
                            "content": "** 허리가 자주 아픈데 다른 병원에서 물리치료만 받았는데, 온누리병원에서도 치료가 가능할까요?**",
                            "answer": "치료는 가능합니다. MRI 검사 결과와 함께 내원하시면 원장님과 상담 후 근본적인 원인을 찾아 해결하는 치료를 받아보실 수 있습니다."
                        },
                        {
                            "content": "** 도수치료가 가능한가요?**",
                            "answer": "네, 원장님 진료 후 필요하다고 판단되면 도수치료가 가능하며, 주사 치료는 환자 상태에 따라 시행합니다."
                        },
                        {
                            "content": "** 측만증 치료는 얼마나 소요되나요?**",
                            "answer": "측만증 치료는 1시간 정도 소요되며, 예약제로 운영됩니다. 운동치료는 저녁 8시까지 예약 가능하며, 내원 상담 후 치료 여부를 결정합니다."
                        },
                        {
                            "content": "** 아이가 또래보다 키가 작아 걱정하는데, 병원에서 어떤 치료를 받을 수 있나요?**",
                            "answer": "온누리 통증의원에서 예상 키 측정을 위해 키 측정 후 손목 및 상반신 X-ray 촬영을 진행하며, 성장판을 자극하는 운동 및 생활 습관 교정을 통해 아이의 성장에 도움을 드립니다."
                        }]
                    ,
                    required=['content', 'answer']
                )
            )
        )
    }
)
@api_view(['GET'])
def faq_list_view(request):
    """
    클러스터링 기반으로 추출된 대표 질문/답변 FAQ 리스트 반환
    """
    try:
        faqs = load_approved_faqs()
        return Response(faqs, status=200)
    except Exception:
        return Response({"error": "Unable to load approved FAQ drafts."}, status=500)


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
    serializer = ImageRecommendationSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    from .inference import recommend_images_by_question
    results = recommend_images_by_question(serializer.validated_data["content"])
    return Response({"results": results})
