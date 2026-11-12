"""Detailed inference pipeline, stage by stage as datamate/pipeline.py runs it, with the flags of the submitted setup
(scripts/flags.sh). Same look as fig_pipeline.py (style.py). Labels in Vietnamese.

Run: .venv/bin/python docs/figures/fig_pipeline_detail.py  ->  docs/figures/pipeline_detail.{png,pdf,svg}
"""
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch

from style import C, arrow, box, gbox, rich, save, setup

# Some Linux machines carry two Liberation Serif sets; the old one (truetype/liberation/) lacks Vietnamese stacked
# diacritics (ề, ấ, ồ) and matplotlib may pick it for italics. Drop it so every style renders Vietnamese.
fm.fontManager.ttflist = [f for f in fm.fontManager.ttflist if "/truetype/liberation/" not in f.fname]

W, H = 18.4, 10.25
fig, ax = setup(W, H)
B, G, I = {"weight": "bold"}, {"color": C["grey"]}, {"style": "italic"}
M = {"family": "monospace", "size": 10.5}
PAL = {k: (C[k + "_fill"], C[k + "_edge"], C[k + "_text"]) for k in "bgyp"}
PAL["r"] = (C["r_fill"], C["r_edge"], C["r_edge"])
PAL["k"] = ("#E9E9E9", "#595959", C["dark"])


def module(x, y, w, h, key, title, lines, step=None, size=11.5):
    """Gradient module: bold title on top, then one line per entry (a str, or rich() parts)."""
    fill, edge, text = PAL[key]
    gbox(ax, x, y, w, h, fill, edge, r=0.18)
    ax.text(x + w / 2, y + h - 0.33, title, ha="center", va="center", fontsize=14, fontweight="bold", color=text, zorder=5)
    if step:
        ax.text(x + 0.2, y + h - 0.33, step, ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.18", fc=edge, ec="none"))
    for k, line in enumerate(lines):
        yy = y + h - 0.75 - k * 0.3
        if isinstance(line, str):
            ax.text(x + w / 2, yy, line, ha="center", va="center", fontsize=size, color=C["dark"], zorder=5)
        else:
            rich(ax, x + w / 2, yy, line, size=size)


def hflow(xs, w, y, color):
    """Block arrows between consecutive modules of one row (left edges xs, equal width w) at height y."""
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w + 0.05, y, b - 0.05, y, color, width=0.2, head_w=0.4, head_l=0.16)


def elbow(pts, color, ls="--", lw=1.8, label=None, label_at=None, z=1):
    """Polyline connector with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, ls=ls, zorder=z))
    if label:
        rich(ax, *label_at, label, size=12.5, color=color)


def band(y, h, title, sub, color):
    box(ax, 0.12, y, W - 0.24, h, "none", color, lw=1.1, r=0.25, z=0, ls=(0, (5, 4)))
    rich(ax, 0.4, y + h - 0.3, [(title, {**B, "size": 15.5, "color": color}), ("   " + sub, {**I, "color": C["grey"]})], ha="left")


# ---- title ---------------------------------------------------------------------------------------------------------------------
rich(ax, W / 2, H - 0.35, [("DataMate", {**B, "size": 20}), ("  —  pipeline suy luận chi tiết", {"size": 20})])
rich(ax, W / 2, H - 0.78, [("LLM chỉ viết SQL; ", I), ("DuckDB", {**I, **B}), (" tính ra con số. Không đủ phiếu thì ", I),
                           ("abstain", {**I, **B, "color": C["r_edge"]}), (" (0 điểm) thay vì đoán sai (−1).", I)], size=13, color=C["grey"])

# ---- row A: once per shop ------------------------------------------------------------------------------------------------------
YA, HA = 6.55, 2.2
band(YA - 0.25, HA + 0.75, "A · Một lần cho mỗi tiệm", "views.py · debts.py — nháp bởi LLM, kiểm trên dữ liệu", C["b_edge"])
wa, xa = 2.68, [0.4 + i * 2.98 for i in range(6)]
ya = YA - 0.05
module(xa[0], ya, wa, 2.1, "b", "Nạp dữ liệu", [
    [("ingest.py", M), (" → DuckDB", {})], "đoán header, chuẩn hoá ngày,", "số tiền, tên hàng (canon)", "+ profile từng cột"], step="1")
module(xa[1], ya, wa, 2.1, "p", "Nháp view", [
    [("K = 8", B), (" câu SELECT", {})], "model gốc, thinking 6k", "map bảng thô → cột chuẩn", [("order_id, qty, amount…", M)]], step="2")
module(xa[2], ya, wa, 2.1, "y", "Kiểm trên dữ liệu", [
    [("catalog·time·split·qty", M)], [("nháp lỗi → ", {}), ("sửa ≤ 2 lượt", B)], "chưa có đa số →", "thêm 1 vòng 8 nháp"], step="3")
module(xa[3], ya, wa, 2.1, "g", "Chọn theo đa số", [
    "cùng fingerprint kết quả", [("≥ K/4", B), (" nháp đồng ý mới dùng", {})], "qty lệch giá > 3× →", "quy đổi đơn vị theo giá"], step="4")
module(xa[4], ya, wa, 2.1, "p", "Lớp công nợ", [
    "cùng công thức: 8 nháp,", "kiểm, sửa, bỏ phiếu", [("debt_at(day)", M)], [("debt_new · debt_repaid", M)]], step="5")
module(xa[5], ya, wa, 2.1, "k", "Ngữ cảnh tiệm", [
    "README + SCHEMA", "+ CANONICAL VIEWS", [("v_sales · v_orders", M)], "+ macro công nợ"], step="6")
hflow(xa, wa, ya + 1.05, C["b_edge"])

# ---- row B: per question -------------------------------------------------------------------------------------------------------
YB, HB = 1.95, 2.6
band(YB - 0.45, HB + 1.55, "B · Mỗi câu hỏi", "pipeline.py: build_chats → sample → decide", C["p_edge"])
wb, xb = 2.2, [0.4 + i * 2.5 for i in range(7)]
yb = YB
module(xb[0], yb, wb, 2.1, "k", "Câu hỏi", [
    [("“doanh thu ", I), ("tháng trước", {**I, **B, "color": C["p_text"]}), ("?”", I)],
    [("as_of_date", M)], "+ rulebook đầy đủ", [("không cắt bớt luật", G)]], step="1")
module(xb[1], yb, wb, 2.1, "y", "Gợi ý", [
    "khung ngày theo as_of_date", "giá trị khớp tên trong câu", "gợi ý theo dạng câu", [("kỳ ngoài dữ liệu? ", {}), ("(oor)", M)]],
       step="2", size=11)
module(xb[2], yb, wb, 2.1, "b", "2 prompt", [
    [("SYSTEM", M), (" + rulebook", {})], [("prompt 2: + ", {}), ("4 ví dụ", B)], "few-shot giống câu hỏi", [("mở đầu: ", {}), ("Notes:", M)]],
       step="3", size=11)
module(xb[3], yb, wb, 2.1, "p", "8 mẫu SQL", [
    [("Qwen3.5-9B", B), (" + LoRA", {})], "vLLM fp8, T = 0.7", "4 mẫu × 2 prompt", [("TYPE: … + SQL", M)]], step="4", size=11)
module(xb[4], yb, wb, 2.1, "g", "Chạy DuckDB", [
    "chỉ đọc, 1 luồng, 6 GB", "thiếu TYPE → điền TYPE", [("lỗi → ", {}), ("sửa ≤ 2 lượt", B)], "có lỗi kèm gợi ý sửa"], step="5", size=11)
module(xb[5], yb, wb, 2.1, "y", "Bỏ phiếu", [
    "gộp đáp án như scorer", "câu công nợ: chỉ mẫu", "gọi macro được bầu", [("share", B), (" = đồng ý / tổng", {})]], step="6", size=11)
hflow(xb[:6], wb, yb + 1.05, C["p_edge"])

# outputs of the vote
xo, wo = xb[6], wb
outs = [("g", "share ≥ 0.5", "→ trả lời", 3.25), ("r", "thấp · 0 · rỗng", "→ abstain", 2.38), ("k", "kỳ ngoài dữ liệu", "→ unanswerable", 1.51)]
for key, a_, b_, yy in outs:
    fill, edge, text = PAL[key]
    gbox(ax, xo, yy, wo, 0.72, fill, edge, lw=1.4, r=0.14)
    ax.text(xo + wo / 2, yy + 0.5, a_, ha="center", va="center", fontsize=11.5, color=text, zorder=5)
    ax.text(xo + wo / 2, yy + 0.22, b_, ha="center", va="center", fontsize=12.5, fontweight="bold", color=text, zorder=5)
    elbow([(xb[5] + wb + 0.05, yb + 1.05), (xo - 0.18, yb + 1.05), (xo - 0.18, yy + 0.36), (xo - 0.03, yy + 0.36)], PAL[key][1], ls="-", lw=1.5)

# thinking loop: vote -> extra thinking samples
lx0, lx1, ly = xb[5] + wb / 2, xb[3] + wb / 2, yb + 2.1 + 0.55
elbow([(lx0, yb + 2.12), (lx0, ly), (lx1, ly), (lx1, yb + 2.14)], C["loop"])
rich(ax, (lx0 + lx1) / 2, ly + 0.25, [("share < 0.75", {**B, **I}), (" hoặc ", I), ("câu công nợ", {**B, **I}), ("  →  ", {}),
                                       ("+4 mẫu thinking", {**B, **I}), (" (16k token), bỏ qua khi chạy chậm tiến độ", I)], size=12, color=C["loop"])

# shop context (A6) feeds the prompts (B3)
cx = xa[5] + wa / 2
elbow([(cx, ya - 0.04), (cx, YB + 3.15), (xb[2] + wb / 2, YB + 3.15), (xb[2] + wb / 2, yb + 2.14)], C["b_edge"], ls="-", lw=1.6)
rich(ax, 11.6, YB + 3.37, [("ngữ cảnh tiệm dùng chung cho mọi câu của tiệm", I)], size=12, color=C["b_edge"])

# ---- row C: the Docker run -----------------------------------------------------------------------------------------------------
YC = 0.25
items = [([("predict.sh", M)], "1 tiến trình / GPU, chia tiệm"),
         ([("--deadline", M)], "tự dừng ở 5 h 50 (giới hạn 6 h)"),
         ([("submission.jsonl", M)], "ghi lại sau mỗi tiệm"),
         ([("logs/<id>.json", M)], "thinking · code · output")]
rich(ax, 0.4, YC + 0.42, [("C · Docker", {**B, "size": 15.5, "color": C["g_edge"]})], ha="left")
wc, xc = 3.55, [2.35 + i * 4.0 for i in range(4)]
for x, (head, sub) in zip(xc, items):
    gbox(ax, x, YC, wc, 0.84, C["g_fill"], C["g_edge"], lw=1.3, r=0.14)
    rich(ax, x + wc / 2, YC + 0.58, head, size=12)
    ax.text(x + wc / 2, YC + 0.26, sub, ha="center", va="center", fontsize=11.5, color=C["dark"], zorder=5)
hflow(xc, wc, YC + 0.42, C["g_edge"])

save(fig, "pipeline_detail")
