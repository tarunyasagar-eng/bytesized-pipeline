"""Build a 1280x720 YouTube thumbnail from a LoRA image: punchy colour grade, big outlined text, optional arrow.
usage: python pipeline/thumbnail.py image.png out.jpg --text "BEFORE THE|FRIDGE?!" [--side left|right] [--arrow]"""
import argparse, os
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, "assets", "fonts", "LilitaOne-Regular.ttf")

def make(src, out, text, side="left", arrow=False, colors=("#ffffff", "#ffd166")):
    im = Image.open(src).convert("RGB")
    s = max(1280 / im.width, 720 / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - 1280) // 2, (im.height - 720) // 2
    im = im.crop((l, t, l + 1280, t + 720))
    im = ImageEnhance.Color(im).enhance(1.25)
    im = ImageEnhance.Contrast(im).enhance(1.12)
    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    d = ImageDraw.Draw(im, "RGBA")
    # darken text side for legibility
    x0 = 0 if side == "left" else 640
    for i in range(640):
        a = int(150 * (1 - i / 640)) if side == "left" else int(150 * (i / 640))
        d.line([(x0 + i, 0), (x0 + i, 720)], fill=(0, 0, 0, a))
    lines = text.split("|")
    size = 150
    f = ImageFont.truetype(FONT, size)
    while max(f.getlength(l) for l in lines) > 600 and size > 60:
        size -= 6; f = ImageFont.truetype(FONT, size)
    total_h = len(lines) * size * 1.05
    y = (720 - total_h) / 2
    for i, line in enumerate(lines):
        w = f.getlength(line)
        x = 50 if side == "left" else 1280 - 50 - w
        d.text((x + 8, y + 10), line, font=f, fill=(0, 0, 0, 160))
        d.text((x, y), line, font=f, fill=colors[i % len(colors)], stroke_width=12, stroke_fill="#1e1a1d")
        y += size * 1.05
    if arrow:
        ax = 640 if side == "left" else 560
        d.line([(ax - 20, 560), (ax + 140, 420)], fill="#e63946", width=34)
        d.polygon([(ax + 175, 385), (ax + 95, 405), (ax + 160, 470)], fill="#e63946")
    im.save(out, quality=95)
    print("thumbnail ->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--text", required=True)
    ap.add_argument("--side", default="left"); ap.add_argument("--arrow", action="store_true")
    a = ap.parse_args()
    make(a.src, a.out, a.text, a.side, a.arrow)
