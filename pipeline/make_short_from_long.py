"""Make a vertical Short from already-generated long-form images (no GPU).
projects/<short>/short.json:
  {"title": "...", "voice": "en-US-BrianNeural", "src": "../clock_story/scenes_qwen",
   "hook": ["LINE 1", "LINE 2"], "top_title": "...", "keywords": [...],
   "segments": [{"text": "...", "imgs": ["c035_0_a.png", ...]}, ...]}
Steps: script.json -> tts.py (vo.wav, words.json) -> edit.json (every image used, segment split evenly, hard cuts,
one camera move per segment) -> music_curious.py.  Then: python pipeline/assemble.py projects/<short>
usage: python pipeline/make_short_from_long.py projects/<short> [--rate +12%]"""
import argparse, json, os, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("--rate", default="+12%"); a = ap.parse_args()
P = lambda f: os.path.join(a.proj, f)
S = json.load(open(P("short.json")))
for seg in S["segments"]:
    for im in seg["imgs"]:
        assert os.path.exists(os.path.join(a.proj, S["src"], im)), im
json.dump({"title": S["title"], "voice": S.get("voice", "en-US-BrianNeural"),
           "segments": [{"text": s["text"]} for s in S["segments"]]}, open(P("script.json"), "w"), indent=1)
py = sys.executable; here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([py, os.path.join(here, "tts.py"), a.proj, "--rate", a.rate], check=True)
Wd = json.load(open(P("words.json"))); seg_t = Wd["seg_t"] + [Wd["total"] + 0.5]
MOVES = ["push", "drift_r", "drift", "drift_l", "pull"]
shots = []
for i, seg in enumerate(S["segments"]):
    t0, t1 = seg_t[i], seg_t[i + 1]; n = len(seg["imgs"])
    for k, im in enumerate(seg["imgs"]):
        shots.append({"img": im, "at": round(t0 + k * (t1 - t0) / n, 3), "move": "punch" if i == 0 and k == 0 else MOVES[i % len(MOVES)],
                      "group": i, **({"transition": "cut"} if k or i else {})})
E0 = {"layout": "panel", "panel_y": 520, "scene_dir": S["src"], "output": "final_short.mp4", "captions": True,
      "caption_style": {"y": 1520, "size": 108, "max_w": 980, "words": 3, "highlight": False, "pop": True, "stroke": 22},
      "music_volume": 0.12, "hook": {"lines": S["hook"], "until": 2.6}, "top_title": S.get("top_title", ""),
      "keywords": S.get("keywords", []), "shots": shots, "overlays": S.get("overlays", [])}
json.dump(E0, open(P("edit.json"), "w"), indent=1)
subprocess.run([py, os.path.join(here, "music_curious.py"), a.proj], check=True)
print(len(shots), "shots, avg", round(seg_t[-1] / len(shots), 2), "s, length", round(Wd["total"], 1), "s")
