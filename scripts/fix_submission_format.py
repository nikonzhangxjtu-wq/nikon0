"""快速修复 submission_v1.csv 的 ret 格式。

旧格式：json.dumps([answer, images]) → '["answer...", ["img1"]]'
新格式：无图 → answer；有图 → answer, ["img1", "img2"]
"""

import json
from pathlib import Path

INPUT = Path("submission_v1.csv")
OUTPUT = Path("submission_v1_fixed.csv")

if not INPUT.exists():
    raise FileNotFoundError(f"找不到 {INPUT.resolve()}")

import pandas as pd

df = pd.read_csv(INPUT)
fixed_count = 0

for idx, row in df.iterrows():
    raw = str(row["ret"])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ⚠️ id={row['id']} JSON 解析失败，跳过")
        continue

    if not isinstance(parsed, list) or len(parsed) != 2:
        print(f"  ⚠️ id={row['id']} 格式异常: {parsed}")
        continue

    answer, images = parsed[0], parsed[1]
    if images and isinstance(images, list) and len(images) > 0:
        df.at[idx, "ret"] = f"{answer}, {json.dumps(images, ensure_ascii=False)}"
    else:
        df.at[idx, "ret"] = answer
    fixed_count += 1

df.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"已处理 {fixed_count}/{len(df)} 条，输出: {OUTPUT.resolve()}")
