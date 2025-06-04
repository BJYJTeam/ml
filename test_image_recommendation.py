from ai_app.inference import recommend_images_by_question

question = "보조기를 언제부터 착용해야 하나요? X-ray 검사를 했는데 25도 정도라고 합니다."
recommended = recommend_images_by_question(question)

for i, img in enumerate(recommended, 1):
    print(f"{i}. {img['description']} — Tags: {', '.join(img['tags'])}")