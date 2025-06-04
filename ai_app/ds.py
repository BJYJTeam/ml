import json

with open('post_comments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("총 댓글 수:", len(data))
print("첫 댓글:", data[0])