import json
from collections import Counter

# ==== JSON 파일 불러오기 ====
# 파일 경로 지정
JSON_PATH = "/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Dialogue_Inference/dataset/original_formatted/dev.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# data는 리스트 형태라고 가정
print(f"총 데이터 개수: {len(data)}")

# ==== category 항목 집계 ====
# category 값만 뽑아서 리스트 만들기
categories = [item["category"] for item in data if "category" in item]

# Counter로 카테고리별 개수 세기
category_counts = Counter(categories)

# ==== 결과 출력 ====
for cat, count in category_counts.items():
    print(f"{cat}: {count}")
