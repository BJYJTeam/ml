from django.urls import path
from .views import answer, doctor_answer_view, extract_keywords, faqs, similar_questions


urlpatterns = [
    path('answer', answer),
    path("doctor-answer", doctor_answer_view),
    path('extract-keywords', extract_keywords),
    path('faqs', faqs),
    path('similar-questions', similar_questions),
]