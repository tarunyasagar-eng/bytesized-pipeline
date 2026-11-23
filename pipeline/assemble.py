"""Assemble a Short from LoRA scene images: camera motion, transitions, captions, overlays, voice + music.
usage: python pipeline/assemble.py projects/<name>          (reads edit.json, words.json, vo.wav, music.wav)
       python pipeline/assemble.py projects/<name> --preview 1.0 8.5 16.5   (writes preview PNGs)"""
import io, json, math, os, subprocess, sys
import cairosvg
from PIL import Image, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, "assets", "fonts", "LilitaOne-Regular.ttf")
FPS = 30
INK = "#1e1a1d"

proj = sys.argv[1]
P = lambda f: os.path.join(proj, f)
E = json.load(open(P("edit.json")))
W, H = (1920, 1080) if E.get("format") == "long" else (1080, 1920)
CAPTIONS = E.get("captions", True)
MUSIC_VOL = E.get("music_volume", 0.15)
D = json.load(open(P("words.json")))
WORDS = D["words"]
TOTAL = D["total"] + 0.5
KEYWORDS = set(E.get("keywords", []))
HOOK = E.get("hook", {})          # {"lines": [...], "until": 2.9}
SCENE_DIR = P(E.get("scene_dir", "scenes_short"))

def find_word(word, after=0.0):
    for w in WORDS:
        if w["t"] >= after - 0.01 and w["w"].lower().strip(".,!?") == word:
            return w["t"]
    return after

# resolve shot start times: number = seconds, "word:<w>@<after>" = word time
def resolve(v):
    if isinstance(v, (int, float)):
        return float(v)
    w, _, after = v.partition("@")
    return find_word(w.split(":")[1], float(after or 0))
SHOTS = [dict(s, t=resolve(s["at"])) for s in E["shots"]]
SHOTS.sort(key=lambda s: s["t"])
for i, s in enumerate(SHOTS):
    s["end"] = SHOTS[i + 1]["t"] if i + 1 < len(SHOTS) else TOTAL
OVERLAYS = [dict(o, t=resolve(o["at"]), until=resolve(o["until"]) if "until" in o else None) for o in E.get("overlays", [])]

CHUNKS, cur = [], []
for k, w in enumerate(WORDS):
    cur.append(k)
    gap = WORDS[k + 1]["t"] - (w["t"] + w["d"]) if k + 1 < len(WORDS) else 9
    if len(cur) >= 3 or gap > 0.18 or k == len(WORDS) - 1:
        CHUNKS.append(cur); cur = []

def ease(x):
    x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def pop(x):
    x = max(0.0, min(1.0, x)); return 1 + 0.25 * math.sin(math.pi * x) * (1 - x) if x < 1 else 1.0
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
_FC = {}
def font(sz):
    if sz not in _FC: _FC[sz] = ImageFont.truetype(FONT, sz)
    return _FC[sz]

def label(x, y, txt, size=90, fill="#ffffff", stroke=INK, sw=16, anchor="middle", rot=0, scale=1.0, opacity=1.0):
    tf = f'translate({x:.1f},{y:.1f}) rotate({rot}) scale({scale:.3f})'
    base = f'font-family="Lilita One" font-size="{size}" text-anchor="{anchor}"'
    return (f'<g transform="{tf}" opacity="{opacity:.2f}"><text {base} fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round">{esc(txt)}</text>'
            f'<text {base} fill="{fill}">{esc(txt)}</text></g>')

# ---------------- images + camera ----------------
_IMG = {}
def shot_image(s):
    key = s["img"]
    if key not in _IMG:
        im = Image.open(os.path.join(SCENE_DIR, key)).convert("RGB")
        sc = max(W / im.width, H / im.height) * 1.14          # headroom for camera moves
        _IMG[key] = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    return _IMG[key]

MOVES = {  # (zoom_start, zoom_end, pan_x -1..1 start->end, pan_y start->end)
    "push": (1.0, 1.12, (0, 0), (0, 0)), "pull": (1.12, 1.0, (0, 0), (0, 0)),
    "left": (1.06, 1.06, (0.8, -0.8), (0, 0)), "right": (1.06, 1.06, (-0.8, 0.8), (0, 0)),
    "up": (1.06, 1.1, (0, 0), (0.7, -0.5)), "down": (1.08, 1.04, (0, 0), (-0.6, 0.6)),
    "punch": (1.0, 1.18, (0, 0), (0, -0.2)),
}
def camera_frame(s, gt):
    im = shot_image(s)
    k = ease((gt - s["t"]) / max(0.1, s["end"] - s["t"]))
    z0, z1, (px0, px1), (py0, py1) = MOVES[s.get("move", "push")]
    z = z0 + (z1 - z0) * k
    if s.get("move") == "punch":                             # fast punch-in then hold
        z = z0 + (z1 - z0) * ease((gt - s["t"]) / 0.35)
    base = min(im.width / W, im.height / H) / 1.14          # source px per output px at zoom 1 (leaves pan headroom)
    cw, ch = W * base / z, H * base / z
    cw, ch = min(cw, im.width), min(ch, im.height)
    px, py = px0 + (px1 - px0) * k, py0 + (py1 - py0) * k
    cx = im.width / 2 + px * (im.width - cw) / 2
    cy = im.height / 2 + py * (im.height - ch) / 2
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    return im.resize((W, H), Image.BILINEAR, box=box)

# ---------------- overlays ----------------
def captions(gt):
    if not CAPTIONS or gt < HOOK.get("until", 0):
        return ""
    active = None
    for ci, ch in enumerate(CHUNKS):
        st = WORDS[ch[0]]["t"]
        en = WORDS[CHUNKS[ci + 1][0]]["t"] if ci + 1 < len(CHUNKS) else TOTAL
        if st <= gt < en:
            active = (ch, st)
    if not active:
        return ""
    ch, st = active
    cy = 1225
    for o in OVERLAYS:
        if o.get("caption_y") and o["t"] <= gt < (o["until"] or TOTAL):
            cy = o["caption_y"]
    words = [WORDS[k]["w"].upper().rstrip(",.") for k in ch]
    size = 118
    def widths(sz): f = font(sz); return [f.getlength(w) for w in words], f.getlength(" ")
    ws, sp = widths(size)
    while sum(ws) + sp * (len(ws) - 1) > 960 and size > 70:
        size -= 6; ws, sp = widths(size)
    x = W / 2 - (sum(ws) + sp * (len(ws) - 1)) / 2
    sc = pop((gt - st) / 0.18)
    out = [f'<g transform="translate({W/2},{cy-35}) scale({sc:.3f}) translate({-W/2},{-(cy-35)})">']
    for k, w, wd in zip(ch, words, ws):
        key = w.lower().strip("!?'") in KEYWORDS or w.isdigit()
        out.append(label(x, cy, w, size, "#ffd166" if (gt >= WORDS[k]["t"] and key) else "#ffffff", INK, 24, anchor="start"))
        x += wd + sp
    return "".join(out) + "</g>"

def hook(gt):
    if not HOOK or gt > HOOK["until"]:
        return ""
    fade = 1 - ease((gt - HOOK["until"] + 0.3) / 0.3)
    k = gt / 0.25
    l1, l2 = HOOK["lines"]
    return (f'<rect x="60" y="190" width="960" height="330" rx="40" fill="#e63946" stroke="{INK}" stroke-width="10" opacity="{fade:.2f}" transform="rotate(-2 540 355)"/>'
            + label(540, 330, l1, 130, "#ffffff", INK, 18, rot=-2, scale=pop(k), opacity=fade)
            + label(540, 470, l2, 130, "#ffd166", INK, 18, rot=-2, scale=pop(k - 0.4) if k > 0.4 else 0.001, opacity=fade))

def overlay(o, gt):
    lt = gt - o["t"]
    if lt < 0 or (o["until"] is not None and gt >= o["until"]):
        return ""
    kind = o["type"]
    if kind == "counter":   # {"from":1,"to":6,"prefix":"DAY "}
        span = (o["until"] or TOTAL) - o["t"]
        n = o["from"] + int((o["to"] - o["from"] + 0.999) * min(1, lt / span))
        return (f'<rect x="300" y="236" width="480" height="150" rx="34" fill="#fff" stroke="{INK}" stroke-width="9"/>'
                f'<text x="540" y="342" font-family="Lilita One" font-size="104" text-anchor="middle" fill="{INK}">{esc(o["prefix"])}{min(n, o["to"])}</text>')
    if kind == "bubble":
        sc = pop(lt / 0.35); bob = 6 * math.sin(gt * 3)
        return (f'<g transform="translate(540,{430+bob:.0f}) scale({sc:.3f})">'
                f'<path d="M-260,110 L-150,260 L-120,110 Z" fill="#fff" stroke="{INK}" stroke-width="8" stroke-linejoin="round"/>'
                f'<rect x="-400" y="-115" width="800" height="230" rx="46" fill="#fff" stroke="{INK}" stroke-width="8"/>'
                f'<rect x="-270" y="100" width="140" height="18" fill="#fff"/>'
                f'<text y="40" font-family="Lilita One" font-size="112" text-anchor="middle" fill="#d62828">{esc(o["text"])}</text></g>')
    if kind == "chapter":    # {"kicker":"REASON 3","text":"BURNOUT"} slides in top-left for ~3s
        dur = (o["until"] or o["t"] + 3.2) - o["t"]
        k_in, k_out = ease(lt / 0.35), 1 - ease((lt - dur + 0.4) / 0.4)
        x = -700 + 760 * k_in - 760 * (1 - k_out)
        f = font(96); w = max(f.getlength(o["text"]) + 80, 420)
        return (f'<g transform="translate({x:.0f},90)"><rect x="0" y="0" width="{w:.0f}" height="190" rx="22" fill="#e63946" stroke="{INK}" stroke-width="8"/>'
                f'<rect x="0" y="-52" width="{font(52).getlength(o["kicker"])+50:.0f}" height="70" rx="16" fill="#ffd166" stroke="{INK}" stroke-width="7"/>'
                f'<text x="25" y="0" font-family="Lilita One" font-size="52" fill="{INK}">{esc(o["kicker"])}</text>'
                + label(40, 130, o["text"], 96, "#ffffff", INK, 16, anchor="start") + '</g>')
    if kind == "tag":        # big label that pops in
        return label(o.get("x", 540), o.get("y", 360), o["text"], o.get("size", 130), o.get("fill", "#ffd166"), INK, 22, scale=pop(lt / 0.3), rot=o.get("rot", 0))
    if kind == "title":      # {"lines":[["THE","#ffd166",110],["DANCING","#fff",190],...],"sub":[...]}
        out = []
        y = o.get("y", 470)
        for j, (txt, col, sz) in enumerate(o["lines"]):
            kk = lt / 0.4 - j * 0.3
            if kk > 0:
                out.append(label(540, y, txt, sz, col, INK, 22, scale=pop(kk)))
            y += sz * 0.95
        for j, (txt, col, x, at) in enumerate(o.get("sub", [])):
            ta = resolve(at)
            if gt >= ta:
                out.append(label(x, y + 40, txt, 110, col, INK, 14, scale=pop((gt - ta) / 0.4)))
        return "".join(out)
    return ""

def overlay_png(gt):
    body = hook(gt) + "".join(overlay(o, gt) for o in OVERLAYS) + captions(gt)
    if not body:
        return None
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>'
    return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode(), output_width=W, output_height=H))).convert("RGBA")

def vignette():
    if not hasattr(vignette, "img"):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><defs><radialGradient id="v" cx="0.5" cy="0.5" r="0.75">'
               f'<stop offset="0.6" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.45"/></radialGradient></defs>'
               f'<rect width="{W}" height="{H}" fill="url(#v)"/></svg>')
        vignette.img = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))).convert("RGBA")
    return vignette.img

def frame(gt):
    i = max(j for j, s in enumerate(SHOTS) if s["t"] <= gt) if gt >= SHOTS[0]["t"] else 0
    s = SHOTS[i]
    img = camera_frame(s, gt)
    xf = 0.18                                              # quick crossfade into each new shot
    if i > 0 and gt - s["t"] < xf and SHOTS[i].get("transition", "fade") == "fade":
        prev = camera_frame(SHOTS[i - 1], gt)
        img = Image.blend(prev, img, ease((gt - s["t"]) / xf))
    elif i > 0 and SHOTS[i].get("transition") == "flash" and gt - s["t"] < 0.1:
        img = Image.blend(img, Image.new("RGB", (W, H), "white"), 0.6 * (1 - (gt - s["t"]) / 0.1))
    img = img.convert("RGBA")
    img.alpha_composite(vignette())
    ov = overlay_png(gt)
    if ov:
        img.alpha_composite(ov)
    return img.convert("RGB")

if __name__ == "__main__":
    if "--preview" in sys.argv:
        for t in sys.argv[sys.argv.index("--preview") + 1:]:
            frame(float(t)).resize((540, 960)).save(P(f"prev_{t}.png"))
        sys.exit()
    out = P(E.get("output", "final.mp4"))
    n = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-i", P("vo.wav"), "-i", P("music.wav"), "-filter_complex",
                           f"[1:a]loudnorm=I=-14:TP=-1.5:LRA=11,apad=pad_dur=1[vo];[2:a]volume={MUSIC_VOL}[mu];"
                           "[vo][mu]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]",
                           "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for k in range(n):
        ff.stdin.write(frame(k / FPS).tobytes())
        if k % 150 == 0:
            print(f"{k}/{n}", flush=True)
    ff.stdin.close(); ff.wait()
    print("wrote", out)
