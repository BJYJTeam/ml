import importlib
import sys
import types
from unittest.mock import Mock

from rest_framework.test import APIRequestFactory

from ai_app.llm import LLMProviderError


def load_views(monkeypatch):
    fake_inference = types.ModuleType("ai_app.inference")
    fake_inference.generate_ai_intern_answer = Mock(return_value={"content": "답변"})
    fake_inference.generate_doctor_draft = Mock(return_value={"content": "답변"})
    fake_inference.extract_keywords = Mock(return_value={"tag": []})
    fake_inference.find_similar_posts = Mock(return_value=[])
    fake_inference.recommend_images_by_question = Mock(return_value=[])
    monkeypatch.setitem(sys.modules, "ai_app.inference", fake_inference)
    sys.modules.pop("ai_app.views", None)
    views = importlib.import_module("ai_app.views")
    monkeypatch.setitem(sys.modules, "ai_app.views", views)
    return views, fake_inference


def test_similar_posts_rejects_missing_post_id_before_retrieval(monkeypatch):
    views, inference = load_views(monkeypatch)

    response = views.similar_posts_view(
        APIRequestFactory().post(
            "/api/api/similar-posts/", {"title": "제목", "content": "본문"}, format="json"
        )
    )

    assert response.status_code == 400
    assert "post_id" in response.data
    inference.find_similar_posts.assert_not_called()


def test_ai_answer_rejects_empty_content_before_llm_generation(monkeypatch):
    views, inference = load_views(monkeypatch)

    response = views.ai_answer_view(
        APIRequestFactory().post(
            "/api/api/ai-answer/",
            {"post_id": "1", "title": "제목", "content": "   "},
            format="json",
        )
    )

    assert response.status_code == 400
    assert "content" in response.data
    inference.generate_ai_intern_answer.assert_not_called()


def test_doctor_draft_rejects_non_list_comments_before_llm_generation(monkeypatch):
    views, inference = load_views(monkeypatch)

    response = views.doctor_draft_view(
        APIRequestFactory().post(
            "/api/api/doctor-draft/",
            {"post_id": "1", "title": "제목", "content": "본문", "comments": "invalid"},
            format="json",
        )
    )

    assert response.status_code == 400
    assert "comments" in response.data
    inference.generate_doctor_draft.assert_not_called()


def test_keyword_and_image_endpoints_reject_invalid_or_overlong_input(monkeypatch):
    views, inference = load_views(monkeypatch)
    factory = APIRequestFactory()

    keyword_response = views.keyword_extraction_view(
        factory.post(
            "/api/api/extract-keywords/",
            {"post_id": "1", "title": ["not", "text"], "content": "본문"},
            format="json",
        )
    )
    image_response = views.recommend_images(
        factory.post("/api/recommend-images/", {"content": "x" * 5001}, format="json")
    )

    assert keyword_response.status_code == 400
    assert image_response.status_code == 400
    inference.extract_keywords.assert_not_called()
    inference.recommend_images_by_question.assert_not_called()


def test_ai_answer_maps_llm_failure_to_a_controlled_service_error(monkeypatch):
    views, inference = load_views(monkeypatch)
    inference.generate_ai_intern_answer.side_effect = LLMProviderError("provider details")

    response = views.ai_answer_view(
        APIRequestFactory().post(
            "/api/api/ai-answer/",
            {"post_id": "1", "title": "제목", "content": "본문"},
            format="json",
        )
    )

    assert response.status_code == 503
    assert response.data == {"error": "AI generation is temporarily unavailable."}


def test_valid_requests_preserve_the_existing_response_contract(monkeypatch):
    views, inference = load_views(monkeypatch)
    factory = APIRequestFactory()

    similar_response = views.similar_posts_view(
        factory.post(
            "/api/api/similar-posts/",
            {"post_id": "1", "title": "제목", "content": "본문"},
            format="json",
        )
    )
    answer_response = views.ai_answer_view(
        factory.post(
            "/api/api/ai-answer/",
            {"post_id": "1", "title": "제목", "content": "본문"},
            format="json",
        )
    )
    doctor_response = views.doctor_draft_view(
        factory.post(
            "/api/api/doctor-draft/",
            {
                "post_id": "1",
                "title": "제목",
                "content": "본문",
                "comments": [{"comment_author": "user", "comment_content": "추가 문의"}],
            },
            format="json",
        )
    )
    keyword_response = views.keyword_extraction_view(
        factory.post(
            "/api/api/extract-keywords/",
            {"post_id": "1", "title": "제목", "content": "본문"},
            format="json",
        )
    )
    image_response = views.recommend_images(
        factory.post("/api/recommend-images/", {"content": "본문"}, format="json")
    )

    assert similar_response.data == {"post_id": "1", "similar_post_ids": []}
    assert answer_response.data == {"content": "답변"}
    assert doctor_response.data == {"content": "답변"}
    assert keyword_response.data == {"tag": []}
    assert image_response.data == {"results": []}
