"""Build edit.json for a long-form video from script.json + words.json + picks.json.
picks.json: {"g001": ["g001_a.png", "g001_c.png"], ...}  (best 1-2 images per shot, best first)
Long segments get both picks (a cut halfway); chapter cards + number callouts are added automatically.
usage: python pipeline/build_edit_long.py projects/<name>"""
import json, os, re, sys
proj = sys.argv[1]; P = lambda f: os.path.join(proj, f)
S = json.load(open(P("script.json"))); Wd = json.load(open(P("words.json")))
shots = json.load(open(P("shots.json"))); picks = json.load(open(P("picks.json")))
seg_t = Wd["seg_t"] + [Wd["total"] + 0.5]
by_seg = {}
for sh in shots: by_seg.setdefault(sh["seg"], []).append(sh["id"])
MOVES = ["push", "right", "pull", "left", "up", "push", "down"]
out_shots, overlays, m = [], [], 0
for i, seg in enumerate(S["segments"]):
    t0, t1 = seg_t[i], seg_t[i + 1]
    imgs = [im for sid in by_seg.get(i, []) for im in picks.get(sid, [])]
    if not imgs:
        continue
    use = imgs[:2] if (t1 - t0) > 5.0 and len(imgs) > 1 else imgs[:1]
    for k, im in enumerate(use):
        out_shots.append({"img": im, "at": round(t0 + k * (t1 - t0) / len(use), 3), "move": MOVES[m % len(MOVES)]})
        m += 1
    ch = seg.get("ch")
    if ch and i > 0:
        kicker, _, title = ch.partition(":")
        if not title: kicker, title = "CHAPTER", ch
        overlays.append({"type": "chapter", "kicker": kicker.strip().upper(), "text": title.strip().upper(), "at": t0 + 0.2, "until": t0 + 3.6})
# number callouts: big pop-in text for key figures
CALL = [(r"three point seven nine billion", "$3.79 BILLION"), (r"eight hundred million dollars on day one", "$800M DAY ONE"),
        (r"two hundred million copies", "200M+ COPIES"), (r"hundred-hour weeks", "100-HOUR WEEKS?"),
        (r"Ninety clips", "90 LEAKED CLIPS"), (r"ninety million views", "90M VIEWS / 24H"),
        (r"four hundred and seventy-five million", "475M VIEWS"), (r"thirty-two and a half million", "32.5M PRE-SOLD"),
        (r"^Thirteen years\.", "13 YEARS"), (r"five times", "~5x GTA 5")]
for i, seg in enumerate(S["segments"]):
    for pat, txt in CALL:
        if re.search(pat, seg["text"], re.I):
            overlays.append({"type": "tag", "text": txt, "at": seg_t[i] + 0.3, "until": seg_t[i + 1] - 0.1, "x": 960, "y": 930, "size": 120})
            break
E0 = {"format": "long", "captions": False, "music_volume": 0.10, "scene_dir": "scenes_long",
      "output": "final_long.mp4", "keywords": [], "shots": out_shots, "overlays": overlays}
json.dump(E0, open(P("edit.json"), "w"), indent=1)
print(len(out_shots), "cuts,", len(overlays), "overlays, avg cut", round(seg_t[-1] / max(1, len(out_shots)), 2), "s")
