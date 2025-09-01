data_map = [
    #{"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Dialogue_Inference/dataset/original_formatted/dev.json", 
    # "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/inference_dev.json"},
    #{"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Dialogue_Inference/dataset/original_formatted/test.json", 
    # "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/inference_test.json"},
    {"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Dialogue_Summarization/dataset/original_formatted/dev.json", 
     "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/summarization_dev.json"},
    {"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Dialogue_Summarization/dataset/original_formatted/test.json", 
     "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/summarization_test.json"},
    {"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Dialogue_Inference/dataset/original_formatted/train.json", 
     "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/inference_train.json"},
    {"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Dialogue_Summarization/dataset/original_formatted/train.json", 
     "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/summarization_train.json"},
    {"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Inappropriate_Detection/dataset/original_formatted/train.json", 
     "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/summarization_train.json"},
    #{"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Inappropriate_Detection/dataset/original_formatted/dev.json", 
    # "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/Inappropriate_dev.json"},
    #{"input":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Korean_Inappropriate_Detection/dataset/original_formatted/test.json", 
    # "output":"/Users/taeyoonkwack/Documents/HCLT-KACL-2025/Method/dataset/naive/Inappropriate_test.json"},
]

import os
import re
import json
import pathlib
from typing import List, Tuple, Dict, Any

import torch
from tqdm import tqdm
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------
# 1) 로그인 (필요 시)
# --------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        HF_TOKEN = input("Enter your HF_TOKEN (press Enter to skip): ").strip()
    except Exception:
        HF_TOKEN = ""
if HF_TOKEN:
    hf_login(HF_TOKEN)

# --------------------------
# 2) 모델 로드
# --------------------------
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
eos_id = tokenizer.eos_token_id

# --------------------------
# 3) 프롬프트/파서 유틸
# --------------------------
SYS_PROMPT = (
    "당신은 대화 재작성 전문가입니다.\n"
    "주어진 대화에서 마지막 발화를 재작성하세요.\n"
    "[규칙]\n"
    "- 먼저 주어진 맥락을 이해하세요.\n"
    "- 의미는 유사하게 유지하되, 맞춤법 교정, 표준어 사용, 표현 변환 등으로 더 쉽게 이해할 수 있도록 바꾸세요.\n"
    "- 불완전한 문장은 보완해서 완전한 문장으로 바꾸세요.\n"
    "- 출력은 재작성된 한 줄만, 다른 설명 없이 출력하세요.\n"
)

def parse_dialogue(raw: str) -> List[Tuple[str, str]]:
    """'화자1: ...' 행들을 (speaker, text) 리스트로 파싱."""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    pairs = []
    for ln in lines:
        m = re.match(r"^\s*([^:]+)\s*:\s*(.*)$", ln)
        if m:
            spk, txt = m.group(1).strip(), m.group(2).strip()
            pairs.append((spk, txt))
        else:
            pairs.append(("화자?", ln.strip()))
    return pairs

def build_dialogue_string(pairs: List[Tuple[str, str]]) -> str:
    return "\n".join(f"{spk}: {txt}" for spk, txt in pairs)

def build_user_prompt(context_pairs: List[Tuple[str, str]], target_text: str) -> str:
    """직전 맥락(최대 5개) + 재작성 대상 발화로 유저 프롬프트 구성."""
    dialogue_block = "\n".join(f"{spk}: {txt}" for spk, txt in context_pairs)
    return f"""[dialogue]
{dialogue_block}
[rewrite]
{target_text}
"""

def apply_chat_template(messages: List[Dict[str, str]]) -> torch.LongTensor:
    """토크나이저 템플릿 기반 입력 IDs 생성. 템플릿이 없으면 간단한 fallback 사용."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        # Fallback: 시스템/유저를 단순 연결
        sys_c = messages[0].get("content", "")
        usr_c = messages[1].get("content", "")
        text = f"<<SYS>>\n{sys_c}\n<</SYS>>\n\n{usr_c}\n"
        return tokenizer(text, return_tensors="pt").input_ids

def decode_new_only(output_ids: torch.LongTensor, input_ids: torch.LongTensor) -> str:
    """입력 길이 이후 부분만 디코딩."""
    gen_ids = output_ids[input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def sanitize_one_line(text: str) -> str:
    """모델 출력에서 첫 유효 한 줄만 추출하고 '화자X:' 접두 제거."""
    if not text:
        return ""
    # 첫 번째 비어있지 않은 줄
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            text = s
            break
    # "화자1:" 같은 접두 제거
    text = re.sub(r"^\s*화자[\d?]+\s*:\s*", "", text)
    # 바깥따옴표 제거
    text = text.strip().strip("“”\"'")
    return text.strip()

def length_ratio_ok(generated: str, original: str) -> bool:
    """생성 길이가 원문 대비 (1/3)~(2) 범위인지 검사."""
    def nl(s: str) -> int:
        return len(s.strip())
    o = max(1, nl(original))
    g = nl(generated)
    ratio = g / o
    return (1/3) <= ratio <= 2.0

# --------------------------
# 4) 한 발화 재작성
# --------------------------
MAX_CONTEXT = 5
MAX_NEW_TOKENS = 128

@torch.no_grad()
def rewrite_utterance(context_pairs: List[Tuple[str, str]], target_text: str) -> Tuple[str, bool]:
    """
    context_pairs: 직전 발화들 (최대 5개)
    target_text: 재작성 대상 원문
    return: (최종 대치 텍스트, error_flag)
        error_flag=True면 "[err] {원문}"을 써야 함
    """
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": build_user_prompt(context_pairs, target_text)},
    ]
    try:
        in_ids = apply_chat_template(messages).to(model.device)
        out = model.generate(
            in_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_id,
        )
        gen = decode_new_only(out[0], in_ids)
        gen = sanitize_one_line(gen)

        if not gen or not length_ratio_ok(gen, target_text):
            return f"[err] {target_text}", True
        return gen, False
    except Exception:
        return f"[err] {target_text}", True

# --------------------------
# 5) 파일 단위 처리
# --------------------------
def process_file(in_path: str, out_path: str) -> Dict[str, Any]:
    p_in = pathlib.Path(in_path)
    if not p_in.exists():
        print(f"[WARN] Input not found: {in_path} -> skipped")
        return {"input": in_path, "output": out_path, "status": "missing"}

    with open(p_in, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
        except Exception as e:
            print(f"[ERROR] JSON load failed: {in_path} ({e})")
            return {"input": in_path, "output": out_path, "status": "json_error"}

    if not isinstance(items, list):
        print(f"[ERROR] Root JSON must be a list: {in_path}")
        return {"input": in_path, "output": out_path, "status": "format_error"}

    rewritten = []
    model_calls = 0
    err_cnt = 0

    for ex in tqdm(items, desc=f"Rewriting: {p_in.name}"):
        if not isinstance(ex, dict):
            rewritten.append(ex)
            continue

        dlg = ex.get("dialogue", None)
        if not isinstance(dlg, str) or not dlg.strip():
            # dialogue 없거나 빈 문자열이면 그대로 보존
            rewritten.append(ex)
            continue

        pairs = parse_dialogue(dlg)
        new_pairs = []
        for i in range(len(pairs)):
            start = max(0, i - MAX_CONTEXT)
            ctx = pairs[start:i]
            spk, target = pairs[i]
            new_text, is_err = rewrite_utterance(ctx, target)
            new_pairs.append((spk, new_text))
            model_calls += 1
            if is_err:
                err_cnt += 1

        ex_new = dict(ex)
        ex_new["dialogue"] = build_dialogue_string(new_pairs)
        rewritten.append(ex_new)

    # 출력 디렉토리 보장
    p_out = pathlib.Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)

    summ = {
        "input": in_path,
        "output": out_path,
        "items": len(items),
        "model_calls": model_calls,
        "errors": err_cnt,
        "status": "ok",
    }
    print("[DONE]", summ)
    return summ

# --------------------------
# 6) 전체 실행
# --------------------------
summary = []
for m in data_map:
    summary.append(process_file(m["input"], m["output"]))

print("\n=== Summary ===")
for s in summary:
    print(s)