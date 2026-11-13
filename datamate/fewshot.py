"""Few-shot generator: for each question, the most similar train questions with an SQL that ran to the gold answer and only
reads the canonical views, so it carries over to any shop. Retrieval is IDF-weighted overlap of words and character trigrams on
the question with accents dropped and digits masked, no extra model. Examples come from other shops only."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from .ingest import norm

DEBT = {"debt_at", "debt_new", "debt_repaid"}
CANON = {"v_sales", "v_orders", "v_sales_all", "v_orders_all", "calendar"} | DEBT


def tables_of(sql):
    cte = {x.lower() for x in re.findall(r'(?i)(?:\bwith|,)\s*"?(\w+)"?\s+as\s*\(', sql)}
    return {x.lower() for x in re.findall(r'(?i)\b(?:from|join)\s+"?(\w+)"?', sql)} - cte


def feats(text):
    t = re.sub(r"\d+", "#", norm(text))
    words = [w for w in t.split("_") if w]
    return Counter(words) + Counter(t.replace("_", " ")[i:i + 3] for i in range(max(0, len(t) - 2)))


def skeleton(sql):
    return re.sub(r"'[^']*'|\d+", "?", re.sub(r"\s+", " ", sql)).lower()


class Bank:
    def __init__(self, path):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.f = [feats(r["question"]) for r in self.rows]
        df = Counter(k for f in self.f for k in f)
        n = len(self.rows)
        self.idf = {k: math.log(1 + n / v) for k, v in df.items()}
        self.norm = [math.sqrt(sum((c * self.idf[k]) ** 2 for k, c in f.items())) for f in self.f]

    def retrieve(self, question, business, k=4, debt=False):
        """Top k rows by cosine, other shops only, no two with the same SQL shape. Debt examples only where the shop has the debt tables."""
        f = feats(question)
        nq = math.sqrt(sum((c * self.idf.get(x, 1.0)) ** 2 for x, c in f.items())) or 1.0
        scored = []
        for i, g in enumerate(self.f):
            if self.rows[i]["business_id"] == business or (not debt and tables_of(self.rows[i]["sql"]) & DEBT):
                continue
            s = sum(c * g[x] * self.idf[x] ** 2 for x, c in f.items() if x in g) / (nq * self.norm[i] or 1.0)
            scored.append((s, i))
        out, seen = [], set()
        for s, i in sorted(scored, reverse=True):
            sk = skeleton(self.rows[i]["sql"])
            if sk not in seen:
                seen.add(sk)
                out.append(self.rows[i])
            if len(out) == k:
                break
        return out


def block(rows):
    """The examples for the user message, kept right before the question."""
    if not rows:
        return ""
    text = "EXAMPLES FROM OTHER SHOPS (they show how a question maps to TYPE and SQL on the canonical views; this shop's RULEBOOK, SCHEMA and values decide, never copy a value or date from them):"
    for r in rows:
        text += f"\nQ: {r['question']}\nTYPE: {r['type']}\n```sql\n{r['sql']}\n```"
    return text


def nearest(question, rows, k, text=lambda r: r["question"], floor=0.0):
    """The k rows closest to the question (same features and IDF weighting as Bank, IDF taken over these rows), best first."""
    if not rows or k <= 0:
        return []
    fs = [feats(text(r)) for r in rows]
    df = Counter(x for f in fs for x in f)
    idf = {x: math.log(1 + len(rows) / v) for x, v in df.items()}
    f = feats(question)
    nq = math.sqrt(sum((c * idf.get(x, 1.0)) ** 2 for x, c in f.items())) or 1.0
    scored = []
    for i, g in enumerate(fs):
        ng = math.sqrt(sum((c * idf[x]) ** 2 for x, c in g.items())) or 1.0
        scored.append((sum(c * g[x] * idf[x] ** 2 for x, c in f.items() if x in g) / (nq * ng), i))
    return [rows[i] for sc, i in sorted(scored, reverse=True)[:k] if sc > floor]


def self_block(rows):
    """Questions of this same shop that every sample agreed on, as examples for its harder questions (no gold answer involved)."""
    if not rows:
        return ""
    text = ("SOLVED QUESTIONS OF THIS SHOP (same tables, values and rules as the question below; every sample agreed on them. "
            "Reuse how they read this shop's tables, never their dates or values):")
    for r in rows:
        text += f"\nQ: {r['question']}\nTYPE: {r['type']}\n```sql\n{r['sql']}\n```"
    return text


def rule_rows(rulebook):
    """The rulebook's table rows, each with the heading it sits under: [{"section", "row", "key"}]."""
    out, section = [], ""
    for line in rulebook.splitlines():
        t = line.strip()
        if t.startswith("#"):
            section = t.lstrip("# ").strip()
        elif t.startswith("|") and not re.fullmatch(r"[|\s:-]+", t):
            cells = [c.strip() for c in t.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] not in ("Luật", "Cách nói", "Từ", "Kiểu"):
                out.append({"section": section, "row": t, "key": cells[0] + " " + section})
    return out


def rule_block(question, rules, k):
    """The k rulebook rows whose first cell is closest to the question, repeated next to it (the whole RULEBOOK stays in the system message)."""
    hit = nearest(question, rules, k, text=lambda r: r["key"], floor=0.08)
    if not hit:
        return ""
    return ("RULEBOOK ROWS CLOSEST TO THIS QUESTION (repeated from the RULEBOOK, which applies in full):\n"
            + "\n".join(f"[{r['section']}] {r['row']}" for r in hit))
