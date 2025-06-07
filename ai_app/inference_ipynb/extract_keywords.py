import json
import requests
import pandas as pd
from tqdm import tqdm

# input : post_id, title, content
# output : post_id, {keywords}


API_URL = "http://hanyang-datascience.duckdns.org:5005/run"
API_TOKEN = "z8y7x6w5v4.n1m2l3k4j5.Team4"
HEADERS = {
    'Authorization': API_TOKEN,
    'Content-Type': 'application/json'
}

PROMPT = """다음은 마취통증의학과 병원에 올라온 환자 또는 보호자의 질문입니다.  
질문 내용을 바탕으로 다음 조건에 따라 의학적으로 중요한 핵심 키워드만 추출하세요.

[출력 목적]  
- 추출된 키워드는 의료진이 환자의 주요 증상, 가능 질환, 치료 방향 등을 빠르게 파악하고, 분류 및 검색 기능에 활용됩니다.

[추출 기준]  
1. 통증 부위, 증상의 성격 또는 유발 조건, 의심되는 질환명, 치료 방식만 추출하세요.  
2. 키워드는 Bi-gram(두 단어 조합)을 우선적으로 생성하고, 적절한 조합이 없을 경우에만 Uni-gram 사용 가능.  
3. 의미가 유사하거나 반복되는 표현은 하나로 병합하여 중복 없이 추출하세요.  
   - 특히 증상의 강도 표현(예: 심한, 매우, 약간)은 무시하고 핵심 증상만 남기세요.  
   - 예: "심한 두통", "심하게 아픈 허리" → 각각 "두통", "허리 통증"  
4. 행동이나 조건에 따른 통증이 나타날 경우, 이를 반영한 표현도 키워드로 추출하세요.  
   - 예: "걸으면 허리 통증"  
5. 의학적으로 무의미한 단어(나이, 성별, 감정, 인삿말, 일상행동, 형용 표현 등)는 제외하세요.  
   - 제외 예시: "여자 아이", "쇼핑", "^^", "답변 부탁드립니다"  
6. 반드시 아래 카테고리 중 해당하는 하나를 키워드 목록의 **가장 앞에 포함**하세요.  
   - [증상 및 진단 문의]
   - [치료 및 시술 문의]
   - [예약, 진료, 비용 문의]
   - [생활관리 및 기타 문의]
  

[출력 형식]  
- 쉼표로 구분된 키워드 목록만 출력하세요.  
- 문장이나 설명은 포함하지 마세요.  
- 카테고리는 항상 키워드 목록의 맨 앞에 위치해야 합니다.

—
"Title" : {title}
"Content" : {content}
"""

def generate_keywords_with_gemma(title, content):
    prompt = PROMPT.format(title=title, content=content)
    data = {
        'model': 'gemma3:12b',
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    if response.ok:
        return response.json().get('response', '[없음]')
    else:
        return f"[Error {response.status_code}] {response.text}"

# JSON 파일 로드
with open("qa_DB_tag.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# tag 생성
for item in tqdm(data):
    title = item.get("title", "")
    content = item.get("content", "")
    item["tag"] = generate_keywords_with_gemma(title, content)

# pandas DataFrame으로 변환 후 CSV 저장
df = pd.DataFrame(data)
df.to_csv("qa_DB_tag_json4.csv", index=False, encoding="utf-8-sig")

print("완료: 'qa_DB_tag_json4.csv'로 저장됨")