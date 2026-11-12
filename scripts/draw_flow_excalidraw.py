"""Toàn bộ luồng DataMate -> docs/datamate_flow.excalidraw (mở bằng excalidraw.com) + docs/flow_scene.json (cho trang xem).

Một nguồn hình duy nhất: danh sách hình trừu tượng (box, diamond, text, arrow, frame), xuất ra hai định dạng.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = 1720  # bề rộng vùng vẽ

# màu kiểu Excalidraw: tên -> (nền sáng, viền)
FILL = {"blue": "#a5d8ff", "green": "#b2f2bb", "yellow": "#ffec99", "red": "#ffc9c9", "violet": "#d0bfff",
        "teal": "#96f2d7", "orange": "#ffd8a8", "gray": "#e9ecef", "white": "#ffffff", "none": "transparent"}
LINE = {"ink": "#1e1e1e", "blue": "#1971c2", "green": "#2f9e44", "violet": "#6741d9", "orange": "#e8590c",
        "red": "#e03131", "teal": "#0c8599", "gray": "#868e96"}

shapes = []


def box(x, y, w, h, title, lines=(), fill="white", stroke="ink", dashed=False, tsize=21, dsize=15.5, bold=False):
    shapes.append(dict(kind="box", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, dashed=dashed))
    th = tsize * 1.25
    dh = dsize * 1.3 * len(lines)
    top = y + (h - th - (dh + 6 if lines else 0)) / 2
    shapes.append(dict(kind="text", x=x + 10, y=top, w=w - 20, lines=[title], size=tsize, role="hand", color="ink",
                       align="center"))
    if lines:
        shapes.append(dict(kind="text", x=x + 10, y=top + th + 6, w=w - 20, lines=list(lines), size=dsize,
                           role="body", color="soft", align="center"))
    return (x, y, w, h)


def diamond(cx, cy, w, h, lines, fill="yellow"):
    shapes.append(dict(kind="diamond", x=cx - w / 2, y=cy - h / 2, w=w, h=h, fill=fill, stroke="ink", dashed=False))
    s = 16
    shapes.append(dict(kind="text", x=cx - w / 2 + 30, y=cy - s * 1.3 * len(lines) / 2, w=w - 60, lines=lines, size=s,
                       role="hand", color="ink", align="center"))
    return (cx - w / 2, cy - h / 2, w, h)


def text(x, y, w, lines, size=16, role="body", color="soft", align="left"):
    shapes.append(dict(kind="text", x=x, y=y, w=w, lines=list(lines), size=size, role=role, color=color, align=align))


def arrow(points, dashed=False, color="ink", label=None):
    shapes.append(dict(kind="arrow", points=points, dashed=dashed, stroke=color))
    if label:
        (x0, y0), (x1, y1) = points[0], points[1]
        text((x0 + x1) / 2 - 60, (y0 + y1) / 2 - 24, 120, [label], size=15, role="hand", color=color, align="center")


def frame(y, h, num, title, color, note=None):
    shapes.append(dict(kind="frame", x=0, y=y, w=W, h=h, stroke=color))
    text(24, y + 16, 900, [f"{num}  {title}"], size=28, role="hand", color=color)
    if note:
        text(W - 640, y + 24, 616, [note], size=15, role="body", color="soft", align="right")


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y)


def chain(bs, color="ink"):
    for a, b in zip(bs, bs[1:]):
        (x0, y0), (x1, y1) = right(a), left(b)
        arrow([(x0 + 6, y0), (x1 - 6, y1)], color=color)


# ---------------------------------------------------------------- tiêu đề
text(0, 0, W, ["DataMate · luồng giải pháp từ đầu đến cuối"], size=44, role="hand", color="ink")
text(0, 62, W, ["Zalo AI Challenge 2026 · Track 2   |   Qwen3.5-9B + vLLM + DuckDB   |   "
                "model viết SQL, không tự đoán số   |   public PA 0.680"], size=18, role="body", color="soft")

# ---------------------------------------------------------------- ① dữ liệu
Y = 120
frame(Y, 330, "①", "Dữ liệu đầu vào", LINE["blue"], "mỗi bộ có rulebook.md + calendar.csv + data/ theo từng shop")
tr = box(40, Y + 80, 300, 120, "train.zip", ["3000 câu · 30 shop", "CÓ đáp án (answers.jsonl)"], fill="blue")
pu = box(40, Y + 220, 300, 90, "public_test.zip", ["1000 câu · 14 shop · không đáp án"], fill="gray")
fs = box(440, Y + 70, 330, 100, "Few-shot bank", ["1194 ví dụ: câu hỏi + SQL đã ra đúng", "assets/fewshot_bank.jsonl"],
         fill="green")
sp = box(440, Y + 190, 330, 100, "Tập tự chấm", ["holdout600 (6 shop giữ riêng)", "val300 · smoke6 → chấm PA trên máy"],
         fill="green")
lo = box(870, Y + 70, 330, 100, "Dữ liệu LoRA", ["SQL đúng sinh trên train", "(làm sau, chỉ dùng train)"],
         fill="white", dashed=True)
pr = box(1300, Y + 70, 380, 100, "Private test", ["ban tổ chức chạy trong Docker", "2×RTX 5090 · ≤ 6 giờ · offline"],
         fill="gray", dashed=True)
text(870, Y + 200, 810, ["Luật: không train trên public / private test.", "public chỉ dùng để chạy và nộp lấy điểm."],
     size=16, role="body", color="soft")
arrow([(346, Y + 120), (434, Y + 120)], color=LINE["blue"])
arrow([(346, Y + 150), (434, Y + 240)], color=LINE["blue"])
arrow([(776, Y + 120), (864, Y + 120)], dashed=True, color=LINE["gray"])

# ---------------------------------------------------------------- ② hạ tầng
Y = 490
frame(Y, 220, "②", "Hạ tầng chạy", LINE["violet"], "không cần Google Drive: tải 3 file zip thẳng lên /content")
b1 = box(40, Y + 80, 250, 110, "repo.zip", ["code + few-shot bank", "64 test qua"], fill="violet")
b2 = box(330, Y + 80, 280, 110, "Notebook Colab", ["colab_public_a100.ipynb", "chờ đủ 3 file rồi chạy"], fill="violet")
b3 = box(650, Y + 80, 300, 110, "Setup ≈ 3 phút", ["uv venv · vLLM 0.32", "tải Qwen3.5-9B (rev cố định)"],
         fill="violet")
b4 = box(990, Y + 80, 300, 110, "Tự nhận GPU", ["A100 → bf16", "L4 · G4 · H100 → fp8 + KV fp8"], fill="violet")
b5 = box(1330, Y + 80, 350, 110, "G4: RTX PRO 6000", ["96 GB · 4,28 triệu token KV", "~4000 token/s · ~10 unit/giờ"],
         fill="yellow")
chain([b1, b2, b3, b4, b5], color=LINE["violet"])

# ---------------------------------------------------------------- ③ mỗi shop
Y = 750
frame(Y, 420, "③", "Chuẩn bị mỗi shop (1 lần / shop, trước khi trả lời)", LINE["teal"],
      "bước này bật thinking · chiếm phần lớn thời gian trên GPU yếu")
c1 = box(40, Y + 80, 330, 160, "Nạp dữ liệu", ["CSV · XLSX · SQLite → DuckDB", "tìm dòng tiêu đề, gộp bảng theo kỳ",
                                              "\"1tr2\" \"25k\" \"55.000đ\" → số", "ngày nhiều kiểu, serial Excel"],
         fill="teal")
c2 = box(420, Y + 80, 360, 160, "Soạn view sạch", ["8 bản nháp SQL (thinking)", "chạy thử trên dữ liệu thật",
                                                   "lỗi → sửa tối đa 2 lần", "bỏ phiếu theo kết quả giống nhau"],
         fill="teal")
c3 = box(830, Y + 80, 360, 160, "Lớp công nợ", ["cùng cách: 8 (+8) bản nháp", "ai nợ · nợ bao nhiêu · đã trả",
                                                "không đồng thuận → không dùng"], fill="teal")
c4 = box(1240, Y + 80, 440, 160, "Ngữ cảnh của shop", ["rulebook + schema đã profile", "+ view sạch + lớp công nợ",
                                                       "→ đưa vào prompt mọi câu hỏi"], fill="green")
chain([c1, c2, c3, c4], color=LINE["teal"])
d1 = diamond(600, Y + 330, 300, 120, ["≥ 50% bản nháp", "cùng kết quả?"])
arrow([bottom(c2), (600, Y + 270)], color=LINE["teal"])
box(840, Y + 290, 330, 80, "Vòng 2", ["thêm 8 bản nháp, bỏ phiếu chung"], fill="white", dsize=15)
arrow([(750, Y + 330), (834, Y + 330)], color=LINE["red"], label="không")
arrow([(450, Y + 330), (380, Y + 330), (380, Y + 262)], color=LINE["green"], label=None)
text(300, Y + 296, 80, ["có"], size=16, role="hand", color=LINE["green"], align="center")
text(40, Y + 300, 260, ["Ví dụ: dev_014 view 6/8", "đồng ý → dùng; công nợ 0/16", "→ bỏ lớp công nợ"],
     size=15, role="body", color="soft")
text(1240, Y + 300, 440, ["Đã sửa: bỏ phiếu có quy tắc phá hoà cố định", "+ DuckDB 1 luồng → chạy lại luôn ra 1 kết quả",
                          "(trước: 8 lần chạy ra 8 view khác nhau)"], size=15, role="body", color="soft")

# ---------------------------------------------------------------- ④ mỗi câu
Y = 1210
frame(Y, 420, "④", "Trả lời từng câu (× 1000)", LINE["orange"], "không thinking khi trả lời (think-k 0) · 68 phút trên G4")
q1 = box(40, Y + 80, 230, 130, "Câu hỏi", ["tiếng Việt", "vd: \"doanh thu", "tháng 3 của shop?\""], fill="orange")
q2 = box(310, Y + 80, 280, 130, "4 ví dụ few-shot", ["câu train giống nhất", "chỉ lấy từ shop khác", "(bank ở ①)"],
         fill="green")
q3 = box(630, Y + 80, 280, 130, "8 mẫu SQL", ["2 kiểu system prompt", "viết trên view sạch"], fill="orange")
q4 = box(950, Y + 80, 280, 130, "DuckDB chạy", ["SQL lỗi → model sửa", "(repair)"], fill="orange")
q5 = box(1270, Y + 80, 410, 130, "Gom nhóm kết quả", ["các mẫu ra cùng đáp án = 1 nhóm", "đếm phiếu nhóm lớn nhất"],
         fill="orange")
chain([q1, q2, q3, q4, q5], color=LINE["orange"])
d2 = diamond(1475, Y + 320, 330, 130, ["nhóm lớn nhất", "≥ 50% phiếu?"])
arrow([bottom(q5), (1475, Y + 254)], color=LINE["orange"])
a1 = box(780, Y + 270, 420, 100, "Trả lời", ["định dạng theo kiểu: number · percent · date", "list · table · boolean · string"],
         fill="green")
a3 = box(40, Y + 270, 520, 100, "Bỏ qua (abstain)", ["0 điểm, an toàn hơn đoán sai (−1)"], fill="red")
arrow([(1310, Y + 320), (1206, Y + 320)], color=LINE["green"], label="có")
text(820, Y + 374, 200, ["không → bỏ qua"], size=16, role="hand", color=LINE["red"], align="center")
arrow([(1475, Y + 385), (1475, Y + 400), (300, Y + 400), (300, Y + 376)], dashed=True, color=LINE["red"])

# ---------------------------------------------------------------- ⑤ nộp bài
Y = 1670
frame(Y, 300, "⑤", "Nộp bài và kết quả", LINE["green"], "Colab tự trả máy khi xong → file phải được lấy ra trước")
e1 = box(40, Y + 80, 280, 120, "submission.jsonl", ["1000 dòng", "kiểm tra format: OK"], fill="green")
e2 = box(360, Y + 80, 320, 120, "In ra output ô chạy", ["gzip + base64 + md5", "(sửa sau khi mất file lần L4)"],
         fill="yellow")
e3 = box(720, Y + 80, 300, 120, "Lấy về máy", ["runs/public_base_g4/", "md5 khớp"], fill="green")
e4 = box(1060, Y + 80, 260, 120, "Portal", ["bạn tải file lên"], fill="gray")
e5 = box(1360, Y + 66, 320, 148, "PA 0.680", ["Public base G4", "không LoRA · think-k 0"], fill="green", tsize=40)
chain([e1, e2, e3, e4, e5], color=LINE["green"])
text(40, Y + 222, 1640, ["846 trả lời  ·  763 đúng  ·  83 sai  ·  154 bỏ qua      |      "
                         "shop yếu: dev_014 35/63 · dev_012 35/50 · dev_006 84/111      |      "
                         "cả lượt ≈ 72 phút trên G4 (L4 hôm trước: 5,8 giờ)"], size=17, role="body", color="ink")

# ---------------------------------------------------------------- ⑥ tiếp theo
Y = 2010
frame(Y, 270, "⑥", "Tiếp theo (hạn Docker 16/11)", LINE["red"], "mọi thay đổi đo trên holdout600 trước, không dò điểm public")
f0 = box(40, Y + 80, 260, 140, "0 · Đóng Docker", ["khoá bản 0.680", "kiểm tra ≤ 6 giờ"], fill="red")
f1 = box(330, Y + 80, 260, 140, "1 · Phân tích lỗi", ["chạy holdout600", "mất điểm ở đâu?"], fill="yellow")
f2 = box(620, Y + 80, 290, 140, "2 · 154 câu bỏ qua", ["thinking cho câu", "chưa đồng thuận",
                                                     "sửa dev_014 · 012 · 006"], fill="yellow")
f3 = box(940, Y + 80, 260, 140, "3 · 83 câu sai", ["chỉnh ngưỡng 50%", "sửa định dạng"], fill="yellow")
f4 = box(1230, Y + 80, 200, 140, "4 · LoRA", ["để sau cùng", "(lợi ích nhỏ)"], fill="white", dashed=True)
f5 = box(1460, Y + 80, 220, 140, "Mục tiêu", ["0.72 – 0.75", "(ước tính)"], fill="green", tsize=26)
chain([f0, f1, f2, f3, f4, f5], color=LINE["red"])

# mũi tên nối giữa các phần
for y0, y1 in ((450, 490), (710, 750), (1170, 1210), (1630, 1670), (1970, 2010)):
    arrow([(W / 2, y0 + 4), (W / 2, y1 - 4)], color=LINE["gray"])

H = 2300


# ================================================================ xuất .excalidraw
def color_of(name):
    return LINE.get(name, name) if not name.startswith("#") else name


def excalidraw(shapes):
    rnd = random.Random(7)
    els = []

    def base(t, x, y, w, h, **kw):
        e = dict(id=f"e{len(els)}", type=t, x=x, y=y, width=w, height=h, angle=0, strokeColor="#1e1e1e",
                 backgroundColor="transparent", fillStyle="solid", strokeWidth=2, strokeStyle="solid", roughness=1,
                 opacity=100, groupIds=[], frameId=None, roundness=None, seed=rnd.randint(1, 2 ** 31),
                 version=1, versionNonce=rnd.randint(1, 2 ** 31), isDeleted=False, boundElements=[], updated=1,
                 link=None, locked=False)
        e.update(kw)
        els.append(e)
        return e

    for s in shapes:
        k = s["kind"]
        if k == "frame":
            base("rectangle", s["x"], s["y"], s["w"], s["h"], strokeColor=s["stroke"], strokeStyle="dashed",
                 strokeWidth=2, roughness=1, roundness={"type": 3})
        elif k in ("box", "diamond"):
            base("rectangle" if k == "box" else "diamond", s["x"], s["y"], s["w"], s["h"],
                 backgroundColor=FILL[s["fill"]], strokeColor=color_of(s["stroke"]),
                 strokeStyle="dashed" if s["dashed"] else "solid",
                 roundness={"type": 3} if k == "box" else {"type": 2})
        elif k == "text":
            lh = 1.25
            t = "\n".join(s["lines"])
            col = {"ink": "#1e1e1e", "soft": "#495057"}.get(s["color"], s["color"])
            base("text", s["x"], s["y"], s["w"], s["size"] * lh * len(s["lines"]), strokeColor=col,
                 text=t, originalText=t, fontSize=s["size"], fontFamily=5 if s["role"] == "hand" else 6,
                 textAlign=s["align"], verticalAlign="top", containerId=None, autoResize=False, lineHeight=lh)
        elif k == "arrow":
            (x0, y0) = s["points"][0]
            pts = [[px - x0, py - y0] for px, py in s["points"]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            base("arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys), points=pts, strokeColor=color_of(s["stroke"]),
                 strokeStyle="dashed" if s["dashed"] else "solid", startBinding=None, endBinding=None,
                 startArrowhead=None, endArrowhead="arrow", elbowed=False, lastCommittedPoint=None,
                 roundness={"type": 2} if len(pts) == 2 else None)
    return {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com", "elements": els,
            "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}


scene = excalidraw(shapes)
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/datamate_flow.excalidraw").write_text(json.dumps(scene, ensure_ascii=False, indent=1), encoding="utf-8")
(ROOT / "docs/flow_scene.json").write_text(json.dumps({"W": W, "H": H, "shapes": shapes, "fill": FILL, "line": LINE},
                                                      ensure_ascii=False), encoding="utf-8")
print(len(scene["elements"]), "elements")
