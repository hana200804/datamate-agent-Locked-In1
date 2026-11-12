"""Pipeline in two stages, paper style (same look as fig_pipeline.py): I. once per shop, the LLM drafts the mapping onto canonical
views, tested on the data and voted; II. per question, 8 SQL samples over those views, executed and voted. Labels in Vietnamese.
The SQL on the cards and the vote counts are illustrations, not taken from any shop.

Run: .venv/bin/python docs/figures/fig_pipeline_v2.py  ->  docs/figures/pipeline_v2.{png,pdf,svg}
"""
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from style import C, arrow, box, flow, gbox, gpoly, logo, mix, rich, save, setup

# Some Linux machines carry two Liberation Serif sets; the old one (truetype/liberation/) lacks Vietnamese stacked
# diacritics (ề, ấ, ồ) and matplotlib may pick it for italics. Drop it so every style renders Vietnamese.
fm.fontManager.ttflist = [f for f in fm.fontManager.ttflist if "/truetype/liberation/" not in f.fname]

W, H = 17.6, 9.2
fig, ax = setup(W, H)
B, G, I = {"weight": "bold"}, {"color": C["grey"]}, {"style": "italic"}
M = {"family": "Liberation Mono"}
GAP = 0.12
Y1, H1 = 5.75, 2.45          # stage I modules
Y2, H2 = 1.15, 2.85          # stage II modules
AY1, AY2 = Y1 + 1.2, 2.6     # arrow heights


def title_top(x, text, color, sub=None):
    ax.text(x, Y1 + H1 + 0.55, text, ha="center", va="center", fontsize=15, fontweight="bold", color=color)
    if sub:
        rich(ax, x, Y1 + H1 + 0.22, sub, size=12.5, color=C["grey"], style="italic")


def title_bottom(x, text, color, sub=None):
    ax.text(x, 0.78, text, ha="center", va="center", fontsize=15, fontweight="bold", color=color)
    if sub:
        rich(ax, x, 0.4, sub, size=12.5, color=C["grey"], style="italic")


def mini_table(x, y, w, h, head, name=None, color=None):
    for i in range(4):
        for j in range(3):
            ax.add_patch(Rectangle((x + j * w / 3, y + h - (i + 1) * h / 4), w / 3, h / 4, fc=head if i == 0 else "white",
                                   ec=color or head, lw=0.8, zorder=4))
    if name:
        ax.text(x + w / 2, y - 0.17, name, ha="center", va="center", fontsize=12, color=C["b_text"], fontweight="bold", style="italic", zorder=5)


def check(x, y, color, s=0.1):
    ax.plot([x - s, x - s * 0.3, x + s * 1.1], [y, y - s * 0.8, y + s * 0.9], color=color, lw=2.4, solid_capstyle="round", solid_joinstyle="round", zorder=6)


def cross(x, y, color, s=0.09):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=2.2, solid_capstyle="round", zorder=6)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=2.2, solid_capstyle="round", zorder=6)


def cards(x, y, w, h, lines, size=9.3):
    for k in (2, 1):
        gbox(ax, x + 0.1 * k, y + 0.1 * k, w, h, C["g_fill"] if size else C["p_fill"], C["g_edge"], lw=1.2, r=0.15, z=2)
    box(ax, x, y, w, h, "white", C["g_edge"], lw=1.6, r=0.15, z=3)
    step = min(0.3, (h - 0.35) / max(1, len(lines) - 1))
    for k, seg in enumerate(lines):
        rich(ax, x + 0.14, y + h - 0.27 - k * step, seg, ha="left", size=size)
    ax.text(x + w + 0.1, y + h + 0.3, "× 8", ha="center", va="center", fontsize=16, fontweight="bold", color=C["g_text"])


def dots(x0, y0, colors, r=0.1, dx=0.28, dy=0.3):
    for k, c in enumerate(colors):
        ax.add_patch(Circle((x0 + (k % 4) * dx, y0 - (k // 4) * dy), r, fc=c, ec="white", lw=1.0, zorder=5))


GR, YE, GY = C["g_edge"], C["gold"], "#BDBDBD"

# ===== stage tags =================================================================================================================
for (yy, tag, sub, col) in [(Y1 + H1 + 0.55, "I · Mỗi tiệm", "một lần, trước mọi câu hỏi", C["b_edge"]),
                            (4.72, "II · Mỗi câu hỏi", "dùng lại lớp dữ liệu của tiệm", C["p_edge"])]:
    rich(ax, 0.3, yy, [(tag, {**B, "size": 15, "color": col})], ha="left")
    rich(ax, 0.3, yy - 0.33, [(sub, {**I, "color": C["grey"]})], ha="left", size=12)

# ===== stage I: once per shop ====================================================================================================
# shop files
logo(ax, "excel.png", 0.45, 6.85, 0.6)
ax.add_patch(Polygon([[1.22, 6.85], [1.7, 6.85], [1.7, 7.3], [1.58, 7.45], [1.22, 7.45]], closed=True, fc="white", ec=C["grey"], lw=1.3, zorder=4))
ax.text(1.46, 7.08, "CSV", ha="center", va="center", fontsize=10, fontweight="bold", color=C["g_text"], zorder=5)
logo(ax, "sqlite.png", 0.5, 6.22, 0.46)
ax.text(1.12, 5.95, "File của tiệm", ha="center", va="center", fontsize=13.5, fontweight="bold", color=C["dark"])
ax.text(1.12, 5.68, "README, bảng lộn xộn", ha="center", va="center", fontsize=11, color=C["grey"], style="italic")

# 1 DuckDB raw tables
XA1, WA1 = 2.55, 2.1
gbox(ax, XA1, Y1, WA1, H1, C["b_fill"], C["b_edge"], r=0.28)
logo(ax, "duckdb.png", XA1 + 0.25, Y1 + 1.68, 0.55, rounded=0.11)
ax.text(XA1 + 0.95, Y1 + 1.95, "DuckDB", ha="left", va="center", fontsize=15, fontweight="bold", color=C["b_text"], zorder=5)
for k in range(2):
    mini_table(XA1 + 0.2 + k * 0.92, Y1 + 0.62, 0.78, 0.72, "#A6A6A6", color=C["grey"])
ax.text(XA1 + WA1 / 2, Y1 + 0.33, "bảng thô  t01 … tNN", ha="center", va="center", fontsize=11.5, color=C["b_text"], style="italic", zorder=5)
title_top(XA1 + WA1 / 2, "Nạp dữ liệu", C["b_text"], [("header", B), (", ngày, số tiền", {})])

# 2 view drafts
XA2, WA2 = 5.3, 3.15
for k in (2, 1):
    gbox(ax, XA2 + 0.1 * k, Y1 + 0.1 * k, WA2, H1 - 0.2, C["p_fill"], C["p_edge"], lw=1.2, r=0.15, z=2)
box(ax, XA2, Y1, WA2, H1 - 0.2, "white", C["p_edge"], lw=1.6, r=0.15, z=3)
vl = [[("-- README + SCHEMA + RULEBOOK", {**M, "color": C["grey"]})],
      [("SELECT", {**M, "color": C["p_text"], "weight": "bold"}), (" h.so_hd   AS ", M), ("order_id", {**M, "color": C["b_text"], "weight": "bold"}), (",", M)],
      [("       h.ngay    AS ", M), ("order_date", {**M, "color": C["b_text"], "weight": "bold"}), (",", M)],
      [("       c.sl*u.hs AS ", M), ("qty", {**M, "color": C["b_text"], "weight": "bold"}), (",", M)],
      [("       c.tien    AS ", M), ("amount", {**M, "color": C["b_text"], "weight": "bold"}), (", …", M)],
      [("FROM", {**M, "color": C["p_text"], "weight": "bold"}), (" t02 h JOIN t03 c …", M)]]
for k, seg in enumerate(vl):
    rich(ax, XA2 + 0.14, Y1 + H1 - 0.5 - k * 0.33, seg, ha="left", size=9.6)
ax.text(XA2 + WA2 + 0.12, Y1 + H1 + 0.08, "× 8", ha="center", va="center", fontsize=16, fontweight="bold", color=C["p_text"])
title_top(XA2 + WA2 / 2 + 0.1, "Nháp ánh xạ", C["p_text"], [("LLM viết", B), (" 8 SELECT sang cột chuẩn", {})])

# 3 data tests
XA3, WA3 = 9.2, 1.95
gbox(ax, XA3, Y1, WA3, H1, C["y_fill"], C["y_edge"], r=0.28)
tests = [("catalog", True), ("time", True), ("split", True), ("qty", False)]
for k, (name, ok) in enumerate(tests):
    yy = Y1 + H1 - 0.45 - k * 0.38
    (check if ok else cross)(XA3 + 0.38, yy, GR if ok else C["r_edge"])
    ax.text(XA3 + 0.65, yy, name, ha="left", va="center", fontsize=12, family="Liberation Mono", color=C["dark"], zorder=5)
rich(ax, XA3 + WA3 / 2, Y1 + 0.5, [("lỗi → ", {"color": C["r_edge"]}), ("sửa ≤ 2 lượt", {**B, "color": C["y_text"]})], size=11.5)
ax.text(XA3 + WA3 / 2, Y1 + 0.22, "chạy thật trên dữ liệu", ha="center", va="center", fontsize=10.5, color=C["y_text"], style="italic", zorder=5)
title_top(XA3 + WA3 / 2, "Kiểm thử", C["y_text"], [("không", B), (" tin lời model", {})])

# 4 fingerprint vote
XA4, WA4 = 11.9, 1.7
gbox(ax, XA4, Y1, WA4, H1, C["g_fill"], C["g_edge"], r=0.28)
dots(XA4 + 0.43, Y1 + H1 - 0.45, [GR] * 5 + [YE] + [GY] * 2)
rich(ax, XA4 + WA4 / 2, Y1 + 1.3, [("5 / 8", {"weight": "bold", "color": C["g_text"]}), (" trùng", {"color": C["g_text"]})], size=12.5)
ax.text(XA4 + WA4 / 2, Y1 + 1.0, "fingerprint kết quả", ha="center", va="center", fontsize=10.5, color=C["g_text"], style="italic", zorder=5)
rich(ax, XA4 + WA4 / 2, Y1 + 0.5, [("dùng khi ", {"color": C["g_text"]}), ("≥ K/4", {"weight": "bold", "color": C["r_edge"]})], size=11.5)
ax.text(XA4 + WA4 / 2, Y1 + 0.22, "không thì bỏ view", ha="center", va="center", fontsize=10.5, color=C["grey"], style="italic", zorder=5)
title_top(XA4 + WA4 / 2, "Chọn đa số", C["g_text"], [("8 nháp", B), (", tự nhất quán", {})])

# 5 canonical layer
XA5, WA5 = 14.35, 3.0
gbox(ax, XA5, Y1, WA5, H1, C["b_fill"], C["b_edge"], r=0.28)
mini_table(XA5 + 0.3, Y1 + 1.05, 1.0, 0.95, C["b_edge"], "v_sales")
mini_table(XA5 + 1.7, Y1 + 1.05, 1.0, 0.95, C["b_edge"], "v_orders")
box(ax, XA5 + 0.55, Y1 + 0.17, 1.9, 0.38, "white", C["b_edge"], lw=1.1, r=0.14, z=4)
ax.text(XA5 + 1.5, Y1 + 0.36, "debt_at(day)", ha="center", va="center", fontsize=12, style="italic", color=C["b_text"], zorder=5)
title_top(XA5 + WA5 / 2, "Lớp dữ liệu chuẩn", C["b_text"], [("view", B), (" + ", {}), ("macro công nợ", B)])

arrow(ax, 1.95 + GAP, AY1, XA1 - GAP, AY1, C["b_edge"])
arrow(ax, XA1 + WA1 + GAP, AY1, XA2 - GAP, AY1, C["p_edge"])
arrow(ax, XA2 + WA2 + 0.2 + GAP, AY1, XA3 - GAP, AY1, C["y_edge"])
arrow(ax, XA3 + WA3 + GAP, AY1, XA4 - GAP, AY1, C["g_edge"])
arrow(ax, XA4 + WA4 + GAP, AY1, XA5 - GAP, AY1, C["b_edge"])

# ===== stage II: per question ====================================================================================================
# question
box(ax, 0.3, 2.05, 2.05, 1.35, "white", C["dark"], lw=1.5, r=0.3)
rich(ax, 1.325, 3.1, [("“dthu ", I), ("tháng trước", {**I, **B, "color": C["p_text"]})], size=13.5)
rich(ax, 1.325, 2.77, [("bnhiu?”", I)], size=13.5)
rich(ax, 1.325, 2.35, [("as_of_date ", G), ("03-10", {"color": C["y_text"], "weight": "bold"})], size=11)
title_bottom(1.325, "Câu hỏi", C["dark"], [("tiếng Việt chat", {})])

# 1 prompt
XB1, WB1 = 2.95, 2.2
ax.add_patch(Polygon([[XB1, Y2], [XB1 + WB1, Y2], [XB1 + WB1, Y2 + H2 - 0.3], [XB1 + WB1 - 0.28, Y2 + H2], [XB1, Y2 + H2]], closed=True, fc="white", ec=C["dark"], lw=1.6, zorder=2))
ax.add_patch(Polygon([[XB1 + WB1 - 0.28, Y2 + H2], [XB1 + WB1 - 0.28, Y2 + H2 - 0.3], [XB1 + WB1, Y2 + H2 - 0.3]], closed=True, fc=C["light"], ec=C["dark"], lw=1.2, zorder=3))
parts = [("rulebook", "#E4E4E4", C["dark"]), ("schema + views", C["b_fill"], C["b_text"]), ("gợi ý ngày, giá trị", C["y_fill"], C["y_text"]),
         ("few-shot SQL", C["g_fill"], C["g_text"]), ("câu hỏi", C["p_fill"], C["p_text"])]
for k, (name, fc, tc) in enumerate(parts):
    yy = Y2 + H2 - 0.62 - k * 0.48
    gbox(ax, XB1 + 0.17, yy, WB1 - 0.34, 0.37, fc, mix(tc, 0.35), lw=0.9, r=0.1, z=3)
    ax.text(XB1 + WB1 / 2, yy + 0.185, name, ha="center", va="center", fontsize=11.5, color=tc, zorder=5)
title_bottom(XB1 + WB1 / 2, "Prompt", C["dark"], [("2 bản", B), (", bản 2 có few-shot", {})])

# 2 LLM
XB2, XB2e = 5.85, 7.55
gpoly(ax, [[XB2, Y2], [XB2, Y2 + H2], [XB2e, Y2 + H2 - 0.55], [XB2e, Y2 + 0.55]], C["p_fill"], C["p_edge"])
logo(ax, "qwen_gh.png", XB2 + 0.6, Y2 + 2.0, 0.5)
ax.text(XB2 + 0.85, Y2 + 1.62, "Qwen3.5-9B", ha="center", va="center", fontsize=14.5, fontweight="bold", color=C["p_text"], zorder=5)
ax.text(XB2 + 0.85, Y2 + 1.3, "+ LoRA", ha="center", va="center", fontsize=13.5, color=C["p_text"], zorder=5)
w = logo(ax, "vllm.png", XB2 + 0.22, Y2 + 0.7, 0.28)
ax.text(XB2 + 0.22 + w + 0.05, Y2 + 0.84, "vLLM (fp8)", ha="left", va="center", fontsize=10.5, color=C["p_text"], zorder=5)
title_bottom(XB2 + 0.85, "LLM", C["p_text"], [("≤ 15B", B), (" tham số, offline", {})])

# 3 SQL samples
XB3, WB3 = 8.25, 3.05
sl = [[("Notes: tháng trước = [02-01, 03-01)", {**M, "color": C["grey"]})],
      [("TYPE: ", {**M, "color": C["p_text"]}), ("number", {**M, "color": C["p_text"], "weight": "bold"})],
      [("SELECT ", M), ("SUM(amount)", {**M, "color": C["g_text"], "weight": "bold"})],
      [("FROM ", M), ("v_sales", {**M, "color": C["b_text"], "weight": "bold"})],
      [("WHERE order_date >= '2026-02-01'", M)],
      [("  AND order_date <  '2026-03-01'", M)]]
for k in (2, 1):
    gbox(ax, XB3 + 0.1 * k, Y2 + 0.1 * k, WB3, H2 - 0.25, C["g_fill"], C["g_edge"], lw=1.2, r=0.15, z=2)
box(ax, XB3, Y2, WB3, H2 - 0.25, "white", C["g_edge"], lw=1.6, r=0.15, z=3)
for k, seg in enumerate(sl):
    rich(ax, XB3 + 0.14, Y2 + H2 - 0.6 - k * 0.36, seg, ha="left", size=9.6)
ax.text(XB3 + WB3 + 0.12, Y2 + H2 + 0.05, "× 8", ha="center", va="center", fontsize=16, fontweight="bold", color=C["g_text"])
title_bottom(XB3 + WB3 / 2 + 0.1, "Mẫu SQL", C["g_text"], [("+4 thinking", B), (" khi chưa chắc", {})])

# 4 execute and vote
XB4, WB4 = 12.05, 1.6
c4 = XB4 + WB4 / 2
gbox(ax, XB4, Y2, WB4, H2, C["y_fill"], C["y_edge"], r=0.28)
logo(ax, "duckdb.png", c4 - 0.19, Y2 + 2.28, 0.38, rounded=0.08)
dots(XB4 + 0.38, Y2 + 1.98, [GR] * 6 + [YE] + [GY])
rich(ax, c4, Y2 + 1.36, [("6 / 8", {"weight": "bold", "color": C["g_text"]}), (" đồng ý", {"color": C["y_text"]})], size=12.5)
bx, bw = XB4 + 0.17, WB4 - 0.34
ax.add_patch(Rectangle((bx, Y2 + 0.88), bw, 0.24, fc="white", ec=C["y_edge"], lw=1.0, zorder=5))
ax.add_patch(Rectangle((bx, Y2 + 0.88), bw * 0.75, 0.24, fc=C["g_edge"], ec="none", zorder=5))
ax.plot([bx + bw / 2] * 2, [Y2 + 0.8, Y2 + 1.2], color=C["r_edge"], lw=2.0, zorder=6)
rich(ax, c4, Y2 + 0.6, [("share ≥ ", {"color": C["y_text"]}), ("τ = 0.5", {"weight": "bold", "color": C["r_edge"]})], size=11.5)
ax.text(c4, Y2 + 0.27, "khoá theo scorer", ha="center", va="center", fontsize=10.5, color=C["y_text"], style="italic", zorder=5)
title_bottom(c4, "Chạy và bỏ phiếu", C["y_text"], [("SQL chỉ đọc", B)])

# 5 answer
XB5, WB5 = 14.35, 1.75
c5 = XB5 + WB5 / 2
gbox(ax, XB5, Y2 + 1.55, WB5, 1.3, C["g_fill"], C["g_edge"], lw=1.8, r=0.2)
check(c5, Y2 + 2.5, C["g_edge"], s=0.13)
ax.text(c5, Y2 + 2.05, "48.250.000", ha="center", va="center", fontsize=13, color=C["g_text"], fontweight="bold", zorder=5)
ax.text(c5, Y2 + 1.75, "number", ha="center", va="center", fontsize=10, color=C["grey"], family="Liberation Mono", zorder=5)
gbox(ax, XB5, Y2 + 0.78, WB5, 0.62, C["r_fill"], C["r_edge"], lw=1.4, r=0.16)
ax.text(c5, Y2 + 1.21, "thấp · 0 · rỗng", ha="center", va="center", fontsize=10.5, color=C["r_edge"], zorder=5)
ax.text(c5, Y2 + 0.95, "abstain", ha="center", va="center", fontsize=12.5, color=C["r_edge"], fontweight="bold", zorder=5)
gbox(ax, XB5, Y2, WB5, 0.62, "#E9E9E9", "#595959", lw=1.4, r=0.16)
ax.text(c5, Y2 + 0.43, "kỳ ngoài dữ liệu", ha="center", va="center", fontsize=10.5, color=C["dark"], zorder=5)
ax.text(c5, Y2 + 0.17, "unanswerable", ha="center", va="center", fontsize=12.5, color=C["dark"], fontweight="bold", zorder=5)
title_bottom(c5, "Đáp án", C["g_text"], [("sai bị ", {}), ("−1", B), (", bỏ được 0", {})])

arrow(ax, 2.35 + GAP, AY2, XB1 - GAP, AY2, C["dark"])
arrow(ax, XB1 + WB1 + GAP, AY2, XB2 - GAP, AY2, C["dark"])
arrow(ax, XB2e + GAP, AY2, XB3 - GAP, AY2, C["p_edge"])
arrow(ax, XB3 + WB3 + 0.2 + GAP, AY2, XB4 - GAP, AY2, C["g_edge"])
arrow(ax, XB4 + WB4 + GAP, Y2 + 2.2, XB5 - GAP, Y2 + 2.2, C["g_edge"], width=0.22, head_w=0.44, head_l=0.18)
for yy, col in [(Y2 + 1.09, C["r_edge"]), (Y2 + 0.31, "#595959")]:
    ax.add_patch(FancyArrowPatch((XB4 + WB4 + GAP, yy), (XB5 - GAP, yy), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.5, zorder=3))

# retry loop: vote -> LLM
lx = XB2 + 1.0
ax.plot([c4, c4, lx, lx], [Y2 + H2 + GAP, 4.55, 4.55, Y2 + H2 - 0.1], color=C["loop"], lw=1.7, ls=(0, (4, 2.5)), zorder=1)
ax.add_patch(FancyArrowPatch((lx, Y2 + H2 + 0.1), (lx, Y2 + H2 - 0.32 + GAP), arrowstyle="-|>", mutation_scale=14, color=C["loop"], lw=1.7, zorder=3))
flow(ax, 9.75, 4.78, [[("lỗi SQL", B)], [("sửa ≤ 2 lượt", {})]], size=12.5, ha="right", color=C["loop"], style="italic")
flow(ax, 10.2, 4.78, [[("share < 0.75", B), (" / công nợ", {})], [("+4 thinking", {})]], size=12.5, ha="left", color=C["loop"], style="italic")

# stage I feeds stage II: the canonical layer is the prompt's schema
cx, py = XA5 + WA5 / 2, 5.2
ax.plot([cx, cx, XB1 + WB1 / 2], [Y1 - GAP, py, py], color=C["b_edge"], lw=1.9, zorder=1)
ax.add_patch(FancyArrowPatch((XB1 + WB1 / 2, py), (XB1 + WB1 / 2, Y2 + H2 + GAP), arrowstyle="-|>", mutation_scale=16, color=C["b_edge"], lw=1.9, zorder=3))
rich(ax, 10.2, py + 0.2, [("view và macro đi vào prompt; ", {}), ("mẫu few-shot chỉ đọc view", B)], size=12, color=C["b_edge"], style="italic")

save(fig, "pipeline_v2")
