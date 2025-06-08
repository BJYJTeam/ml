from django.urls import path
from .views import (
    ai_answer_view,
    doctor_draft_view,
    keyword_extraction_view,
    faq_list_view,
    similar_posts_view,
    recommend_images,
)

urlpatterns = [
    path('api/similar-posts/', similar_posts_view, name='similar-posts'),
    path('api/ai-answer/', ai_answer_view, name='ai-answer'),
    path('api/doctor-draft/', doctor_draft_view, name='doctor-draft'),
    path('api/extract-keywords/', keyword_extraction_view, name='extract-keywords'),
    path('api/faqs/', faq_list_view, name='faq-list'),
    path('recommend-images/', recommend_images, name='recommend-images'),
]