import importlib
import sys
import types

from rest_framework.test import APIRequestFactory


def test_faq_endpoint_reads_stored_approved_drafts_without_calling_inference(monkeypatch):
    fake_inference = types.ModuleType("ai_app.inference")
    fake_inference.generate_ai_intern_answer = lambda *args, **kwargs: None
    fake_inference.generate_doctor_draft = lambda *args, **kwargs: None
    fake_inference.extract_keywords = lambda *args, **kwargs: None
    fake_inference.find_similar_posts = lambda *args, **kwargs: None
    fake_inference.recommend_images_by_question = lambda *args, **kwargs: None
    fake_inference.generate_faqs_from_db = lambda: (_ for _ in ()).throw(
        AssertionError("FAQ endpoint must not invoke the inference facade")
    )
    monkeypatch.setitem(sys.modules, "ai_app.inference", fake_inference)
    sys.modules.pop("ai_app.views", None)
    views = importlib.import_module("ai_app.views")
    monkeypatch.setitem(sys.modules, "ai_app.views", views)
    monkeypatch.setattr(
        views,
        "load_approved_faqs",
        lambda: [{"content": "승인 질문", "answer": "승인 답변"}],
    )

    response = views.faq_list_view(APIRequestFactory().get("/api/api/faqs/"))

    assert response.status_code == 200
    assert response.data == [{"content": "승인 질문", "answer": "승인 답변"}]
