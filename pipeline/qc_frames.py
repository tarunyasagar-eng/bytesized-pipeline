"""Automatic clean-frame checks for stickman scenes, run on every generated image before it is used.
Flags: CLONES (2+ near-identical characters side by side), NO_FACE (a stickman head without 2 eyes),
BYTE_x2 (more than one orange cap), plus review sheets (2x2 at 768 px) for a final visual pass.
usage: python pipeline/qc_frames.py <image dir> [--sheets out_dir] [--json report.json]"""
import argparse, json, os
import cv2, numpy as np

def heads(img):
    """Round white stickman heads: bright, low-saturation, roughly circular blobs with a dark outline."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 215) & (hsv[..., 1] < 40)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(white)
    H, W = white.shape; out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.0012 * H * W or a > 0.12 * H * W or not (0.7 < w / max(h, 1) < 1.35): continue
        r = (w + h) / 4; fill = a / (np.pi * r * r)
        if fill < 0.72: continue
        ring = cv2.dilate((lab == i).astype(np.uint8), np.ones((7, 7), np.uint8)) - (lab == i).astype(np.uint8)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (g[ring > 0] < 90).mean() < 0.35: continue          # needs a dark outline
        out.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), r=float(r), mask=(lab == i)))
    return out

def eyes(img, hd):
    """Dark blobs fully inside the head = eyes/mouth (dot eyes)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = hd["x"], hd["y"], hd["w"], hd["h"]
    full = cv2.morphologyEx(hd["mask"].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # fill holes = features
    inner = cv2.erode(full, np.ones((5, 5), np.uint8))[y:y + h, x:x + w]
    dark = ((g[y:y + h, x:x + w] < 110) & (inner > 0)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(dark)
    area = hd["r"] ** 2
    return sum(1 for i in range(1, n) if 0.002 * area < st[i][4] < 0.25 * area)

def crop_sig(img, hd):
    x, y, w, h = hd["x"], hd["y"], hd["w"], hd["h"]
    x0, y0 = max(0, x - w // 3), max(0, y - h // 2); x1, y1 = min(img.shape[1], x + w + w // 3), min(img.shape[0], y + h * 3)
    c = cv2.resize(img[y0:y1, x0:x1], (48, 96)).astype(np.float32)
    return (c - c.mean()) / (c.std() + 1e-6)

def orange_caps(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] > 8) & (hsv[..., 0] < 22) & (hsv[..., 1] > 150) & (hsv[..., 2] > 170)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return sum(1 for i in range(1, n) if st[i][4] > 0.002 * m.size and st[i][2] > 1.3 * st[i][3])   # wide cap-shaped blobs

def check(path):
    img = cv2.imread(path); hs = heads(img); flags = []
    for h in hs:
        h["eyes"] = eyes(img, h)
    if any(h["eyes"] < 2 for h in hs): flags.append("NO_FACE")
    sig = [crop_sig(img, h) for h in hs]; sim = 0
    for i in range(len(hs)):
        same = [j for j in range(len(hs)) if j != i and abs(hs[i]["r"] - hs[j]["r"]) < 0.12 * hs[i]["r"]
                and float((sig[i] * sig[j]).mean()) > 0.72]
        sim = max(sim, len(same) + 1)
    if sim >= 3: flags.append("CLONES")
    elif sim == 2: flags.append("TWINS?")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    org = (hsv[..., 0] > 8) & (hsv[..., 0] < 22) & (hsv[..., 1] > 150) & (hsv[..., 2] > 170)
    capped = 0
    for h in hs:   # an orange cap sits on the top of a head
        x0, x1 = max(0, h["x"] - h["w"] // 4), h["x"] + h["w"] + h["w"] // 4
        y0, y1 = max(0, h["y"] - h["h"] // 2), h["y"] + h["h"] // 3
        if org[y0:y1, x0:x1].mean() > 0.18: capped += 1
    if capped > 1: flags.append("BYTE_x2")
    return dict(heads=len(hs), eyes=[h["eyes"] for h in hs], flags=flags)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--sheets"); ap.add_argument("--json"); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    fs = sorted(f for f in os.listdir(a.dir) if f.endswith(".png") and (not a.only or any(f.startswith(o) for o in a.only)))
    rep = {f: check(os.path.join(a.dir, f)) for f in fs}
    for f, r in rep.items():
        if r["flags"]: print(f, r["flags"], "heads", r["heads"], "eyes", r["eyes"])
    print(f"{sum(1 for r in rep.values() if r['flags'])}/{len(rep)} flagged")
    if a.json: json.dump(rep, open(a.json, "w"), indent=0)
    if a.sheets:
        os.makedirs(a.sheets, exist_ok=True)
        for k in range(0, len(fs), 4):
            tiles = [cv2.resize(cv2.imread(os.path.join(a.dir, f)), (768, 432)) for f in fs[k:k + 4]]
            for t, f in zip(tiles, fs[k:k + 4]):
                cv2.putText(t, f[:12] + " " + " ".join(rep[f]["flags"]), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            while len(tiles) < 4: tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(os.path.join(a.sheets, f"review_{k // 4:03d}.jpg"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
