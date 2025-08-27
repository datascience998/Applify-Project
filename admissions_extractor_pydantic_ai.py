"""
大学招生实体抽取（基于 pydantic-ai）
=================================================
需求：
- 从每行帖子（CSV 的一行，需先将各列合并成一段文本）中抽取：
  1) 大学名称 university
  2) 招生相关术语 terms（严格返回 2 个）
- 预研阶段每次只抽取最多 50 行；对 badcases 做统计与归因，便于后续迭代 prompt；
- prompt 稳定后再批量处理。

使用说明（本地运行）：
1) 安装依赖：
   uv pip install "pydantic_ai>=0.0.15" pydantic pandas logfire python-dotenv
   # 或者：pip install ...

2) 配置模型/密钥（示例为 OpenAI）：
   - 设置环境变量 OPENAI_API_KEY，不要把 key 写进代码或聊天：
     Linux/macOS: export OPENAI_API_KEY=your_key
     Windows(PowerShell): $Env:OPENAI_API_KEY = "your_key"
   - 可选：设置 PYDANTIC_AI_MODEL（默认 'openai:gpt-4o'），例如：
     export PYDANTIC_AI_MODEL=openai:gpt-4o-mini

3) 运行：
   uv run python admissions_extractor_pydantic_ai.py --csv ./posts.csv --out ./out --limit 50
   # 追加从第 51 行开始：
   uv run python admissions_extractor_pydantic_ai.py --csv ./posts.csv --out ./out --limit 50 --start 50

输出：
- out/extractions.csv：抽取结果（row_index, university, term1, term2, ok）
- out/badcases.csv：badcase 详情（含原因、片段）
- out/summary.txt：badcase 统计摘要与改进提示

安全：
- 绝不要把 API Key 写进代码或聊天；仅通过环境变量读取。
- 仅处理研发样本（limit<=50）以控制成本；批量前先迭代 prompt。
"""
from __future__ import annotations

import os
import re
import time
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, conlist, ValidationError

import logfire
from pydantic_ai import Agent

# -------- 日志与模型配置 --------
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
MODEL = os.getenv("PYDANTIC_AI_MODEL", "openai:gpt-4o")

# -------- 受控术语词表（可迭代优化/扩展） --------
CANONICAL_TERMS: List[str] = [
    # 过程/政策类
    "application deadline", "rolling admission", "early decision", "early action", "waitlist",
    "deferral", "transfer", "interview", "orientation", "admissions portal", "fee waiver",
    # 要求/材料类
    "GPA requirement", "SAT", "ACT", "TOEFL", "IELTS", "recommendation letter",
    "personal statement", "resume/CV", "portfolio",
    # 资助/费用类
    "scholarship", "financial aid", "tuition", "deposit",
    # 其他常见招生相关
    "acceptance rate", "campus visit", "major declaration", "international student",
]

# 中英文同义正则（把文本中的缩写/中文表达标准化为上面的 canonical 术语）
TERM_NORMALIZE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bED\b", re.I), "early decision"),
    (re.compile(r"\bEA\b", re.I), "early action"),
    (re.compile(r"\bSAT\b", re.I), "SAT"),
    (re.compile(r"\bACT\b", re.I), "ACT"),
    (re.compile(r"托福|TOEFL", re.I), "TOEFL"),
    (re.compile(r"雅思|IELTS", re.I), "IELTS"),
    (re.compile(r"推荐信|ref(\.?) letter|LOR", re.I), "recommendation letter"),
    (re.compile(r"个人陈述|PS|personal statement", re.I), "personal statement"),
    (re.compile(r"奖学金|scholarship", re.I), "scholarship"),
    (re.compile(r"学费|tuition", re.I), "tuition"),
    (re.compile(r"录取率|acceptance rate", re.I), "acceptance rate"),
    (re.compile(r"滚动录取|rolling admission", re.I), "rolling admission"),
    (re.compile(r"早申|早决定|early decision", re.I), "early decision"),
    (re.compile(r"早行动|early action", re.I), "early action"),
    (re.compile(r"GPA( 要求)?|GPA requirement", re.I), "GPA requirement"),
    (re.compile(r"经济资助|助学金|financial aid", re.I), "financial aid"),
    (re.compile(r"入学(押金|定金)|deposit", re.I), "deposit"),
    (re.compile(r"免(申|报名)?费|fee waiver", re.I), "fee waiver"),
    (re.compile(r"国际学生|international student", re.I), "international student"),
    (re.compile(r"转学|transfer", re.I), "transfer"),
]

# -------- Pydantic 输出模型 --------
class AdmissionEntity(BaseModel):
    university: Optional[str] = Field(
        None, description="官方大学名称；若文本未提及或无法确定，则为 null"
    )
    terms: conlist(str, min_length=2, max_length=2) = Field(
        ..., description="严格返回 2 个招生相关术语（使用下方受控词表的英文规范名）"
    )

# -------- Prompt 生成 --------
INSTRUCTION_HEADER = f"""
You are an information extraction engine for bilingual (Chinese/English) social posts about university admissions.
Return ONLY JSON that matches the provided Pydantic schema. No explanations.

Task:
- Extract the official university name if present; if none, set to null.
- Extract EXACTLY TWO admissions-related terms from the controlled vocabulary below. If more than two are present, choose the two most salient; if fewer, include those found and fill the rest with "N/A".
- Prefer admissions concepts (process/policy, requirements, materials, aid/fees). Ignore department names unless the university name itself is given.
- Normalize synonyms/abbreviations to the canonical forms provided.

Controlled Vocabulary (canonical English surface forms):
{', '.join(CANONICAL_TERMS)}
""".strip()

SCHEMA_HINT = """
Pydantic schema fields:
- university: string or null
- terms: list of 2 strings (exactly 2) from the controlled vocabulary
""".strip()


def build_prompt(post_text: str) -> str:
    return (
        INSTRUCTION_HEADER
        + "\n\n"
        + SCHEMA_HINT
        + "\n\nPost:\n" + post_text.strip()
    )

# -------- 文本预处理 --------
def normalize_terms_from_text(text: str) -> List[str]:
    """可选的轻量正则提示：帮助模型聚焦文本中可能出现的术语。
    返回文本中疑似术语（已规范化）去重后的列表，供 prompt 参考（不强制）。
    """
    candidates: List[str] = []
    for pat, canon in TERM_NORMALIZE_PATTERNS:
        if pat.search(text):
            candidates.append(canon)
    # 去重并保留出现顺序
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:10]


def merge_row_to_text(row: pd.Series) -> str:
    # 将该行所有列转为字符串后用 “ | ” 连接
    return " | ".join(str(x) for x in row.astype(str).tolist())


# -------- badcase 判定 --------
UNI_PATTERNS = [
    re.compile(r"[A-Z][A-Za-z&.\- ]+ University"),
    re.compile(r"[A-Z][A-Za-z&.\- ]+ College"),
    re.compile(r"[\u4e00-\u9fa5]{2,}大学"),
]


def likely_mentions_university(text: str) -> bool:
    return any(p.search(text) for p in UNI_PATTERNS)


BADCASE_RULES = {
    "validation_error": "LLM 输出未能通过 Pydantic 校验（结构或类型不符）",
    "missing_university": "文本疑似出现学校，但未抽取/解析错误",
    "insufficient_terms": "terms 非 2 个或出现非受控术语",
}


def check_badcase(text: str, ent: Optional[AdmissionEntity], err: Optional[Exception]) -> Tuple[bool, str]:
    if err is not None:
        return True, "validation_error"
    if ent is None:
        return True, "validation_error"
    # 规则：若文本像是提到学校，但 university 为 None
    if ent.university in (None, "", "null") and likely_mentions_university(text):
        return True, "missing_university"
    # 规则：terms 必须为 2 个，且在受控词表或为 N/A
    if len(ent.terms) != 2:
        return True, "insufficient_terms"
    ok_terms = 0
    for t in ent.terms:
        if t == "N/A" or t in CANONICAL_TERMS:
            ok_terms += 1
    if ok_terms != 2:
        return True, "insufficient_terms"
    return False, ""


# -------- 抽取执行 --------

def extract_batch(agent: Agent, rows: List[Tuple[int, str]], sleep_s: float = 0.2):
    """对一批（<=50）行执行抽取，返回 (results_df, badcases_df)。"""
    records = []
    bads = []

    for row_idx, text in rows:
        # 可选提示：给模型一个“可能的术语候选”
        hints = normalize_terms_from_text(text)
        prompt = build_prompt(text)
        if hints:
            prompt += "\n\nPotential terms in text (hints, not mandatory):\n- " + ", ".join(hints)

        ent: Optional[AdmissionEntity] = None
        err: Optional[Exception] = None
        try:
            res = agent.run_sync(prompt)
            ent = res.output  # 类型：AdmissionEntity
            usage = res.usage()
        except Exception as e:  # 捕获 LLM 或验证异常
            err = e
            usage = None

        is_bad, reason = check_badcase(text, ent, err)

        if ent is None:
            uni, terms = None, ["N/A", "N/A"]
        else:
            uni, terms = ent.university, ent.terms

        records.append({
            "row_index": row_idx,
            "university": uni,
            "term1": terms[0] if len(terms) > 0 else "",
            "term2": terms[1] if len(terms) > 1 else "",
            "ok": not is_bad,
            "usage": json.dumps(usage) if usage else "",
        })

        if is_bad:
            snippet = text[:400].replace("\n", " ")
            bads.append({
                "row_index": row_idx,
                "reason_code": reason,
                "reason_desc": BADCASE_RULES.get(reason, reason),
                "university": uni,
                "terms": json.dumps(terms, ensure_ascii=False),
                "snippet": snippet,
                "hints": ", ".join(hints),
            })

        time.sleep(sleep_s)  # 简单限速，预研阶段防止速率过高

    return pd.DataFrame(records), pd.DataFrame(bads)


# -------- 总控（读取 CSV & 分块） --------

def run(csv_path: Path, out_dir: Path, limit: int = 50, start: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # 将每行合并成帖子文本
    posts = df.apply(merge_row_to_text, axis=1).tolist()

    end = min(start + limit, len(posts))
    subset = [(i, posts[i]) for i in range(start, end)]

    print(f"Using model: {MODEL}")
    agent = Agent(MODEL, output_type=AdmissionEntity)

    results_df, bad_df = extract_batch(agent, subset)

    # 保存输出
    results_csv = out_dir / "extractions.csv"
    badcases_csv = out_dir / "badcases.csv"
    results_df.to_csv(results_csv, index=False)
    bad_df.to_csv(badcases_csv, index=False)

    # 统计摘要
    total = len(subset)
    ok = int(results_df["ok"].sum())
    bad = total - ok
    by_reason = bad_df["reason_code"].value_counts().to_dict() if not bad_df.empty else {}

    summary_lines = [
        f"总数: {total}",
        f"通过: {ok}",
        f"Badcases: {bad}",
        "按原因统计:",
    ]
    if by_reason:
        for k, v in by_reason.items():
            summary_lines.append(f"- {k}: {v}")
    else:
        summary_lines.append("- 无")

    # 针对常见 badcase 给出下一轮 prompt 优化建议
    tips = []
    if by_reason.get("missing_university", 0) > 0:
        tips.append("在提示中增加：‘如果只出现院系/学院名，请尝试推断上级大学；若无法确定，明确返回 null。’")
    if by_reason.get("insufficient_terms", 0) > 0:
        tips.append("强调：‘terms 必须严格返回 2 个，且只允许使用受控词表；若不足，补 \"N/A\"。’")
    if by_reason.get("validation_error", 0) > 0:
        tips.append("在开头再次强调 ‘Return ONLY JSON that matches schema, no extra text.’ 并给 1-2 个正/反例。")

    summary_lines.append("\n下一轮 Prompt 迭代建议：")
    if tips:
        summary_lines.extend(["- " + t for t in tips])
    else:
        summary_lines.append("- 当前提示较稳定，可扩大样本或引入更难案例。")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n=== 摘要 ===")
    print("\n".join(summary_lines))
    print(f"\n结果已保存至: {results_csv}\nBadcases: {badcases_csv}\n摘要: {out_dir / 'summary.txt'}")


# -------- CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大学招生实体抽取（pydantic-ai）")
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--limit", type=int, default=50, help="每次处理的最大行数（预研阶段 <= 50）")
    parser.add_argument("--start", type=int, default=0, help="起始行号（用于分批处理）")
    args = parser.parse_args()

    run(Path(args.csv), Path(args.out), limit=args.limit, start=args.start)
