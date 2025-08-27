import os
import json
from glob import glob

# ===== 설정 =====
INPUT_FOLDER = "temp"   # 여러 json 파일이 들어있는 폴더
OUTPUT_JSON = "merged_all.json"

# ===== 처리 =====
all_items = []

# 폴더 내의 .json 파일들 탐색
json_files = glob(os.path.join(INPUT_FOLDER, "*.json"))

print(f"발견된 JSON 파일 수: {len(json_files)}")

for path in json_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"[경고] {path} 은 리스트 형식이 아님 → 스킵")
                continue
            all_items.extend(data)
        print(f"  {os.path.basename(path)}: {len(data)}개 항목 추가")
    except Exception as e:
        print(f"[에러] {path}: {e}")

print(f"총 합쳐진 데이터 수: {len(all_items)}")

# ===== 저장 =====
os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_items, f, ensure_ascii=False, indent=2)

print(f"병합 완료 → {OUTPUT_JSON}")
