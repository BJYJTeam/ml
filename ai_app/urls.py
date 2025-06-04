from django.urls import path
from .views import answer, doctor_answer_view, extract_keywords, faqs, similar_questions, recommend_images
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('answer', answer),
    path("doctor-answer", doctor_answer_view),
    path('extract-keywords', extract_keywords),
    path('faqs', faqs),
    path('similar-questions', similar_questions),
    path("recommend-images/", recommend_images),
]