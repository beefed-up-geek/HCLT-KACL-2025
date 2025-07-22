import os
import glob
import json
import re
import pandas as pd

# ── ① MuTual dev 세트가 있는 폴더 ─────────────────────────────
WORK_DIR = "MuTual/data/mutual/train"   # ← 원하는 경로로 바꾸세요
OUTPUT_CSV = "mutual_train.csv"                  # 저장될 CSV 이름
# ─────────────────────────────────────────────────────────────

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
rows = []

for fp in sorted(glob.glob(os.path.join(WORK_DIR, "train_*.txt"))):
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)

    # ── article: 화자 전환 시 줄바꿈 ───────────────────────────
    #  m : ...  f : ...  ↦  m : ...\n f : ...
    segments = re.split(r"\s*(?=[mMfF]\s*:)", d["article"].strip())
    article = "\n".join(s.strip() for s in segments if s)

    # ── options: A. … B. … 형식 ─────────────────────────────
    opts = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(d["options"]))

    rows.append(
        {
            "id": d["id"],
            "article": article,
            "options": opts,
            "answers": d["answers"],
        }
    )

# ── DataFrame → CSV 저장 ───────────────────────────────────
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"▶ CSV 저장 완료: {OUTPUT_CSV}")
