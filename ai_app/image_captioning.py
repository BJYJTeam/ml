from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
#import openai
import os
from dotenv import load_dotenv
import json
#from openai import OpenAI

load_dotenv()

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Set your OpenAI API key (best to load from environment in production)
#openai.api_key = os.getenv("OPENAI_API_KEY")

#if not openai.api_key:
#    raise EnvironmentError("OPENAI_API_KEY not found in environment.")

#client = OpenAI()

def generate_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

#def generate_tags_with_llm(description: str) -> dict:
#    prompt = f"""
#아래 문장은 척추측만증 관련 의료 이미지에 대한 설명입니다.
#이 문장을 기반으로 아래 항목을 생성하세요:
#
#1. 자연어 설명 1줄
#2. 연관 태그 5개 이내 (중복 X, 키워드 형태)
#
#문장: "{description}"
#형식은 JSON으로:
#{{
#  "description": "...",
#  "tags": ["...", "..."]
#}}
#"""
#    response = client.chat.completions.create(
#        model="gpt-4o-mini",
#        messages=[
#            {"role": "system", "content": "당신은 의료 이미지 태깅 전문가입니다."},
#            {"role": "user", "content": prompt}
#        ],
#        temperature=0.7,
#    )
#    result = response.choices[0].message.content
#    try:
#        return json.loads(result)
#    except json.JSONDecodeError:
#        raise ValueError(f"Failed to parse GPT response: {result}")

def generate_image_metadata(image_path: str) -> dict:
    raw_caption = generate_caption(image_path)
    print("Generated caption:", raw_caption)
    # Fallback tag extraction
    words = raw_caption.lower().split()
    filtered_words = [word.strip(".,!?") for word in words if len(word) > 2]
    tags = list(dict.fromkeys(filtered_words))[:5]  # remove duplicates, limit to 5

    metadata = {
        "filename": os.path.basename(image_path),
        "description": raw_caption, # placeholder, update when GPT is available
        "tags": tags,
        "raw_caption": raw_caption  
    }
    print("Fallback metadata:", metadata)
    return metadata