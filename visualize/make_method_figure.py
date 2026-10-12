"""Generates assets/method.svg, the RetHOI method overview used in the README.

    python visualize/make_method_figure.py assets/method.svg assets/method_output_frame.jpg

The output frame (a mesh render of a generated sample) is embedded into the SVG as base64."""
import base64
import math
import sys

OUT, FRAME = sys.argv[1], sys.argv[2]
W, H = 1800, 880
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

C = dict(
    ret="#3B6FB6", ret_bg="#E8F0FB", cross="#1F4E8C",
    gab="#55657A", gab_bg="#E9EDF2",
    con="#E8710A", con_bg="#FDEBDD",
    apdm="#C2185B", apdm_bg="#FBE4EC",
    guide="#16857F", guide_bg="#E0F3F2",
    frozen="#7D838D", frozen_bg="#F1F2F4",
    ink="#1D2330", sub="#59606E", line="#3A4150",
)
SUB = "baseline-shift='sub' font-size='10'"
SUP = "baseline-shift='super' font-size='10'"
parts = []


def add(s):
    parts.append(s)


def text(x, y, s, size=15, weight="normal", color=None, anchor="middle", italic=False):
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color or C["ink"]}" text-anchor="{anchor}"{st}>{s}</text>')


def box(x, y, w, h, fill, stroke, r=10, dash=None, sw=1.6):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def block(x, y, w, h, title, sub=None, kind="ret", ts=15, sw=1.6):
    box(x, y, w, h, C[kind + "_bg"], C[kind], sw=sw)
    cy = y + h / 2 + (5 if sub is None else -4)
    text(x + w / 2, cy, title, ts, "bold", C[kind])
    if sub:
        text(x + w / 2, cy + 19, sub, 12, color=C["sub"])


def path(pts, color=None, dash=None, sw=1.8, head=True):
    color = color or C["line"]
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#ah)"' if head else ""
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{dd}{m}/>')


def panel(x, y, w, h, label, color):
    box(x, y, w, h, "#FFFFFF", color, r=16, dash="7 5", sw=1.8)
    pw = len(label) * 8.3 + 26
    add(f'<rect x="{x + 18}" y="{y - 14}" width="{pw}" height="28" rx="14" fill="{color}"/>')
    text(x + 31, y + 5, label, 14, "bold", "#FFFFFF", anchor="start")


add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# ================================================================ inputs (x 20..240)
IX, IW = 20, 215
cards = {
    "hoi": (95, "Noisy HOI",
            f"x<tspan {SUB}>t</tspan> = {{ x<tspan {SUP}>h</tspan><tspan {SUB}>t</tspan> , x<tspan {SUP}>o</tspan><tspan {SUB}>t</tspan> }}",
            "L×263 human · L×6 object"),
    "text": (255, "Text prompt d", "“Someone is hauling and", "dragging the chairwood …”"),
    "obj": (440, "Object geometry p", "1024 surface points", "(scanned BEHAVE mesh)"),
    "aff": (620, "Noisy affordance",
            f"y<tspan {SUB}>n</tspan> : 8 key joints ×", "(contact point, probability)"),
}
for y, t1, t2, t3 in cards.values():
    box(IX, y, IW, 116, "#FAFBFC", "#B8BEC8", r=12, dash="5 4")
    text(IX + IW / 2, y + 32, t1, 16, "bold")
    text(IX + IW / 2, y + 62, t2, 14, color=C["sub"], italic=True)
    text(IX + IW / 2, y + 86, t3, 12.5, color=C["sub"], italic=True)

# encoders
block(268, 278, 130, 72, "CLIP ❄", "text encoder", "frozen")
block(268, 463, 130, 72, "PointNet++", "object encoder", "frozen")
path([(IX + IW, 314), (268, 314)])
path([(IX + IW, 499), (268, 499)])

# ================================================================ PRDM (x 455..1300, y 40..420)
PX, PY, PW, PH = 455, 40, 845, 385
panel(PX, PY, PW, PH, "PRDM · Parallel Retentive Diffusion Model   (T = 1000 steps)", C["ret"])
HY, OY, BH = 95, 255, 70          # human row / object row
block(490, HY, 125, BH, "RetNet ×4", "human · MDM init")
block(490, OY, 125, BH, "RetNet ×2", "object")
box(665, HY - 5, 175, OY - HY + BH + 10, C["ret_bg"], C["cross"], sw=2.2)
text(752, HY + 25, "Cross-Retention ×2", 15, "bold", C["cross"])
text(752, HY + 47, "bidirectional, O(N)", 12, color=C["sub"])
text(752, HY + 82, f"H ← H + 0.5·Ret(Q<tspan {SUB}>h</tspan>, K<tspan {SUB}>o</tspan>, V<tspan {SUB}>o</tspan>)", 12, color=C["ink"])
path([(735, HY + 100), (735, OY + 12)], C["cross"], sw=2.2)
path([(770, OY + 12), (770, HY + 100)], C["cross"], sw=2.2)
text(752, OY + 38, f"O ← O + 0.5·Ret(Q<tspan {SUB}>o</tspan>, K<tspan {SUB}>h</tspan>, V<tspan {SUB}>h</tspan>)", 12, color=C["ink"])
block(890, HY, 125, BH, "RetNet ×4", "human")
block(1065, HY, 120, BH, "Gabor", "learnable blend", "gab")
block(1065, OY, 120, BH, "Gabor", "learnable blend", "gab")
path([(615, HY + 35), (665, HY + 35)])
path([(615, OY + 35), (665, OY + 35)])
path([(840, HY + 35), (890, HY + 35)])
path([(1015, HY + 35), (1065, HY + 35)])
path([(840, OY + 35), (1065, OY + 35)])
text(1196, HY + 40, f"x̂<tspan {SUP}>h</tspan><tspan {SUB}>0</tspan>", 20, "bold", anchor="start")
text(1196, OY + 40, f"x̂<tspan {SUP}>o</tspan><tspan {SUB}>0</tspan>", 20, "bold", anchor="start")
text(552, HY - 14, f"[ t , CLIP(d) ] token + x<tspan {SUB}>t</tspan>", 12, color=C["sub"])

# retention inset (bottom-left of PRDM)
RY = 372
text(490, RY, f"Multi-scale retention:  Ret(X) = (QK<tspan {SUP}>⊤</tspan> ⊙ D) V ,  "
              f"D<tspan {SUB}>nm</tspan> = γ<tspan {SUP}>|n−m|</tspan>", 14, anchor="start")
text(490, RY + 26, "4 heads · learnable decay γ · SwiGLU FFN · GroupNorm + swish gate", 12.5,
     color=C["sub"], anchor="start")
cols = ["#A9C8F0", "#6F9FDC", "#3B6FB6", "#1F4E8C"]
for k, (gam, col) in enumerate(zip([0.85, 0.90, 0.95, 0.98], cols)):
    pts = " ".join(f"{915 + i * 1.6:.1f},{RY + 26 - 40 * gam ** i:.1f}" for i in range(60))
    add(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
    text(1012, RY - 18 + k * 13.5, f"γ = {gam:.2f}", 11, "bold", col, anchor="start")
text(965, RY + 42, "decay vs. |n−m|", 11, color=C["sub"])

# motion contrastive head (bottom-right of PRDM)
block(1150, 345, 132, 62, "InfoNCE", "motion ↔ text", "con", ts=14)
path([(1040, HY + 35), (1040, 336), (1150, 336)], C["con"], dash="4 3", sw=1.6)
for ty in (HY + 35, OY + 35):
    add(f'<circle cx="1040" cy="{ty}" r="4" fill="{C["con"]}"/>')
text(1048, 205, "pool · MLP", 11.5, color=C["con"], anchor="start")
text(1048, 221, "(human + object)", 11, color=C["con"], anchor="start")

# inputs into PRDM
path([(IX + IW, 153), (430, 153), (430, HY + 45), (490, HY + 45)])
path([(430, 153), (430, OY + 45), (490, OY + 45)])
path([(398, 300), (444, 300), (444, HY + 22), (490, HY + 22)], C["frozen"], dash="4 3")
path([(444, 300), (444, OY + 22), (490, OY + 22)], C["frozen"], dash="4 3")

# ================================================================ APDM (x 455..1300, y 480..735)
AX, AY, AW, AH = 455, 482, 845, 258
panel(AX, AY, AW, AH, "APDM / CSGM · Affordance Prediction Diffusion   (N = 500 steps)", C["apdm"])
block(490, 530, 235, 84, "Transformer encoder", "256 object tokens + 8 contact tokens", "apdm", ts=15)
block(780, 530, 165, 84, f"Linear head", "ŷ₀ ∈ ℝ^(8 × 4)", "apdm")
block(1100, 530, 182, 84, "InfoNCE", "affordance ↔ text", "con", ts=14)
path([(725, 572), (780, 572)])
path([(752, 572), (752, 512), (1040, 512), (1040, 552), (1100, 552)], C["con"], dash="4 3", sw=1.6)
add(f'<circle cx="752" cy="572" r="4" fill="{C["con"]}"/>')
text(900, 506, "pool · MLP", 11, color=C["con"])
text(490, 650, "8 key joints: pelvis, upper back, feet, shoulders, wrists —", 12.5, color=C["sub"], anchor="start")
text(490, 670, "each with a contact probability and a contact point on the object.", 12.5, color=C["sub"], anchor="start")
path([(398, 499), (418, 499), (418, 552), (490, 552)], C["frozen"], dash="4 3")
path([(398, 330), (412, 330), (412, 588), (490, 588)], C["frozen"], dash="4 3")
path([(IX + IW, 678), (452, 678), (452, 604), (490, 604)])

# CLIP text embedding → both InfoNCE heads (gap between the panels)
path([(333, 350), (333, 452), (1216, 452), (1216, 407)], C["frozen"], dash="2 4", sw=1.4)
path([(1216, 452), (1216, 530)], C["frozen"], dash="2 4", sw=1.4)
text(800, 446, "CLIP text embedding (positive pair, τ = 0.07)", 11.5, color=C["frozen"])

# objective
box(490, 696, 792, 34, "#F7F8FA", "#D3D7DE", r=8)
text(886, 718, f"ℒ = ℒ<tspan {SUB}>diffusion</tspan> + 0.1 · ℒ<tspan {SUB}>contrast</tspan> + 0.05 · ℒ<tspan {SUB}>smooth</tspan>"
               f" ,     ℒ<tspan {SUB}>smooth</tspan> = E ‖ x̂<tspan {SUB}>t+1</tspan> − x̂<tspan {SUB}>t</tspan> ‖²", 14)

# ================================================================ guidance (x 1335..1780)
GX, GY, GW, GH = 1340, 40, 440, 820
panel(GX, GY, GW, GH, "Affordance-guided correction", C["guide"])
bx, bw = GX + 30, GW - 60
block(bx, 85, bw, 78, "Contact trajectories", "contact points of ŷ₀ posed by x̂₀ᵒ  →  L × k × 3", "guide", ts=15)
# Gabor bank
gy = 200
box(bx, gy, bw, 190, C["gab_bg"], C["gab"])
text(bx + bw / 2, gy + 28, "Multi-scale Gabor bank (40 filters)", 15, "bold", C["gab"])
text(bx + bw / 2, gy + 50, "λ ∈ {3, 5, 7, 9, 11} frames × θ = kπ/8 · σ = 0.5λ · γ = 0.5", 12, color=C["sub"])
for i, lam in enumerate([3, 5, 7, 9, 11]):
    x0 = bx + 40 + i * 72
    pts = []
    for s in range(-28, 29):
        t = s / 28 * 5.5
        v = math.exp(-t * t / (2 * (0.5 * lam) ** 2)) * 0.5 * (1 + math.cos(2 * math.pi * t / lam))
        pts.append(f"{x0 + s * 1.0:.1f},{gy + 132 - 50 * v:.1f}")
    add(f'<line x1="{x0 - 30}" y1="{gy + 132}" x2="{x0 + 30}" y2="{gy + 132}" stroke="#C5CBD3" stroke-width="1"/>')
    add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{C["gab"]}" stroke-width="2"/>')
    text(x0, gy + 150, f"λ = {lam}", 11, color=C["sub"])
text(bx + bw / 2, gy + 176, f"ỹ = Σ<tspan {SUB}>s,o</tspan> softmax(w<tspan {SUB}>s</tspan>)·softmax(w<tspan {SUB}>o</tspan>) · y<tspan {SUB}>s,o</tspan>",
     13, color=C["gab"], italic=True)
path([(bx + bw / 2, 163), (bx + bw / 2, gy)])
block(bx, 428, bw, 78, "Guidance  ∇ ℒ_guide", "contact + jitter terms · τ₁ = 1 (human), τ₂ = 10 (object)", "guide", ts=15)
path([(bx + bw / 2, gy + 190), (bx + bw / 2, 428)])

# PRDM output → contact trajectories ; affordance → contact trajectories
path([(1232, HY + 35), (bx, HY + 35)], C["ret"])
path([(1232, OY + 35), (1318, OY + 35), (1318, HY + 50), (bx, HY + 50)], C["ret"])
path([(945, 590), (1070, 590), (1070, 640), (1322, 640), (1322, 148), (bx, 148)], C["apdm"])
text(1200, 660, "ŷ₀ (affordance)", 13, "bold", C["apdm"])

# feedback: guidance → x_{t-1}
path([(bx + bw, 467), (GX + GW - 12, 467), (GX + GW - 12, 22), (1160, 22), (1160, PY)], C["guide"], dash="7 4", sw=2.2)
text(1250, 16, "update x<tspan baseline-shift='sub' font-size='10'>t−1</tspan> at every denoising step", 13,
     "bold", C["guide"], italic=True)

# output
with open(FRAME, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
iw, ih = 250, 272
ix, iy = GX + (GW - iw) / 2 + 40, 548
path([(bx + bw / 2, 506), (bx + bw / 2, iy)], C["guide"])
add(f'<image x="{ix}" y="{iy}" width="{iw}" height="{ih}" href="data:image/jpeg;base64,{b64}" '
    f'xlink:href="data:image/jpeg;base64,{b64}"/>')
add(f'<rect x="{ix}" y="{iy}" width="{iw}" height="{ih}" rx="8" fill="none" stroke="#D3D7DE" stroke-width="1.4"/>')
text(GX + 68, iy + 110, "Output", 16, "bold")
text(GX + 68, iy + 132, "SMPL body", 12.5, color=C["sub"])
text(GX + 68, iy + 150, "+ object", 12.5, color=C["sub"])
text(GX + 68, iy + 168, "mesh", 12.5, color=C["sub"])

# legend
lx, ly = 20, 790
items = [("ret", "Retentive (PRDM)"), ("gab", "Gabor smoothing"), ("con", "Contrastive alignment"),
         ("apdm", "Affordance (APDM)"), ("guide", "Guidance"), ("frozen", "Frozen encoder")]
for i, (k, name) in enumerate(items):
    x = lx + (i % 3) * 140
    y = ly + (i // 3) * 30
    box(x, y - 12, 16, 16, C[k + "_bg"], C[k], r=4, sw=1.6)
    text(x + 24, y + 1, name, 12, color=C["sub"], anchor="start")

add("</svg>")
open(OUT, "w").write("\n".join(parts))
print("wrote", OUT)
