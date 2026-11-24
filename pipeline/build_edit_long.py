"""Build edit.json for a long-form video from script.json + words.json + picks.json.
picks.json: {"g001": ["g001_a.png", "g001_c.png"], ...}
Default (SDXL): best 1-2 images per shot, best first; long segments get both picks (a cut halfway).
--sequence (Qwen pose swaps): every listed image is used in order, the segment split evenly between them, hard cuts,
one shared slow camera drift per segment. Keys g001.. map to segment 0.. when there is no shots.json.
Chapter cards are automatic; number callouts come from callouts.json ([["regex", "TEXT"], ...]) if present.
usage: python pipeline/build_edit_long.py projects/<name> [--sequence] [--scene-dir scenes_qwen]"""
import json, os, re, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("--sequence", action="store_true")
ap.add_argument("--scene-dir", default="scenes_long"); a = ap.parse_args()
proj = a.proj; P = lambda f: os.path.join(proj, f)
S = json.load(open(P("script.json"))); Wd = json.load(open(P("words.json")))
picks = json.load(open(P("picks.json")))
seg_t = Wd["seg_t"] + [Wd["total"] + 0.5]
by_seg = {}
if os.path.exists(P("shots.json")):
    for sh in json.load(open(P("shots.json"))): by_seg.setdefault(sh["seg"], []).append(sh["id"])
else:
    for gid in picks: by_seg.setdefault(int(gid[1:]) - 1, []).append(gid)
MOVES = ["drift", "drift_r", "drift", "drift_l"] if a.sequence else ["push", "right", "pull", "left", "up", "push", "down"]
out_shots, overlays, m = [], [], 0
for i, seg in enumerate(S["segments"]):
    t0, t1 = seg_t[i], seg_t[i + 1]
    imgs = [im for sid in by_seg.get(i, []) for im in picks.get(sid, [])]
    if not imgs:
        continue
    if a.sequence:
        for k, im in enumerate(imgs):
            out_shots.append({"img": im, "at": round(t0 + k * (t1 - t0) / len(imgs), 3), "move": MOVES[m % len(MOVES)],
                              "group": i, **({"transition": "cut"} if k else {})})
        m += 1
    else:
        use = imgs[:2] if (t1 - t0) > 5.0 and len(imgs) > 1 else imgs[:1]
        for k, im in enumerate(use):
            out_shots.append({"img": im, "at": round(t0 + k * (t1 - t0) / len(use), 3), "move": MOVES[m % len(MOVES)]})
            m += 1
    ch = seg.get("ch")
    if ch and i > 0:
        kicker, _, title = ch.partition(":")
        if not title: kicker, title = "CHAPTER", ch
        overlays.append({"type": "chapter", "kicker": kicker.strip().upper(), "text": title.strip().upper(), "at": t0 + 0.2, "until": t0 + 3.6})
# number callouts: big pop-in text for key figures (per project)
CALL = json.load(open(P("callouts.json"))) if os.path.exists(P("callouts.json")) else []
for i, seg in enumerate(S["segments"]):
    for pat, txt in CALL:
        if re.search(pat, seg["text"], re.I):
            overlays.append({"type": "tag", "text": txt, "at": seg_t[i] + 0.3, "until": seg_t[i + 1] - 0.1, "x": 960, "y": 760, "size": 120})
            break
E0 = {"format": "long", "captions": True,
      "caption_style": {"y": 1000, "size": 64, "max_w": 1500, "words": 6, "highlight": False, "pop": False, "stroke": 14, "gap": 2.0}, "music_volume": 0.10, "scene_dir": a.scene_dir,
      "output": "final_long.mp4", "keywords": [], "shots": out_shots, "overlays": overlays}
json.dump(E0, open(P("edit.json"), "w"), indent=1)
print(len(out_shots), "cuts,", len(overlays), "overlays, avg cut", round(seg_t[-1] / max(1, len(out_shots)), 2), "s")
