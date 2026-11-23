"""Build a fuller YouTube description from script.json (+ words.json for chapter timestamps).
Writes projects/<name>/upload.json (title, description, tags) — edit by hand if needed.
usage: python pipeline/describe.py projects/<name> [--short]"""
import argparse, json, os
def ts(t):
    t = int(t); return f"{t//60}:{t%60:02d}"
ap = argparse.ArgumentParser(); ap.add_argument("project"); ap.add_argument("--short", action="store_true")
a = ap.parse_args()
S = json.load(open(os.path.join(a.project, "script.json")))
W = json.load(open(os.path.join(a.project, "words.json"))) if os.path.exists(os.path.join(a.project, "words.json")) else {}
m = S.get("meta", {})
lines = [m.get("hook_line", ""), "", m.get("summary", "")]
if m.get("bullets"):
    lines += ["", "In this video:"] + [f"• {b}" for b in m["bullets"]]
if not a.short and W.get("seg_t"):
    chap = [(W["seg_t"][i], s["ch"]) for i, s in enumerate(S["segments"]) if s.get("ch")]
    if chap and chap[0][0] > 0.5:
        chap[0] = (0, chap[0][1])
    lines += ["", "Chapters:"] + [f"{ts(t)} {c}" for t, c in chap]
if m.get("sources"):
    lines += ["", "Sources:"] + [f"• {s}" for s in m["sources"]]
lines += ["", m.get("cta", "Subscribe to ByteSized AI for more strange-but-true stories, explained in minutes."),
          "", "Illustrations are original stickman art made with our own AI art model. Facts checked at time of upload.",
          "", " ".join(m.get("hashtags", ["#history", "#facts"]))]
desc = "\n".join(l for l in lines).strip()
up = {"title": S["title"] + (" #Shorts" if a.short and "#Shorts" not in S["title"] else ""), "description": desc[:4900],
      "tags": m.get("tags", []), "categoryId": m.get("categoryId", 27)}
json.dump(up, open(os.path.join(a.project, "upload.json"), "w"), indent=1, ensure_ascii=False)
print(desc); print(f"\n[{len(desc)} chars]")
