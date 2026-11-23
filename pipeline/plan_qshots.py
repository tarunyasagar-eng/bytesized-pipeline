"""Turn script.json shot codes (B:/S:/E:) into Kaggle qshots, adding pose/zoom swaps so no image holds longer than --max-hold s.
usage: python pipeline/plan_qshots.py projects/<name> [--max-hold 2.6] [--parts 2]
writes qshots_partN.json and picks.json (segment -> ordered image list)."""
import json, os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("--max-hold", type=float, default=2.6); ap.add_argument("--parts", type=int, default=2)
a = ap.parse_args(); P = lambda f: os.path.join(a.proj, f)
S = json.load(open(P("script.json"))); D = json.load(open(P("words.json")))
seg_t = D["seg_t"] + [D["total"]]
EXTRA = ["same scene, the camera moves a little closer to the main subject, everything else identical",
         "same scene and camera, the main character now looks straight at the viewer with raised eyebrows",
         "same scene and camera, the main character gestures with one open hand as if explaining",
         "same scene, slightly wider camera angle showing more of the background, everything else identical"]
shots, picks, k = [], {}, 0
for i, seg in enumerate(S["segments"]):
    dur = seg_t[i + 1] - seg_t[i]
    codes = list(seg["shots"])
    while dur / len(codes) > a.max_hold:
        codes.append("E: " + EXTRA[k % len(EXTRA)]); k += 1
    ids = []
    for j, c in enumerate(codes):
        kind, prompt = c.split(":", 1)
        sid = f"c{i:03d}_{j}"
        sh = {"id": sid, "prompt": prompt.strip(), "seg": i}
        if kind == "B": sh["ref"] = "byte"
        elif kind == "S": sh["ref"] = "stick"
        else: sh["from"] = shots[-1]["id"]
        shots.append(sh); ids.append(f"{sid}_a.png")
    picks[f"g{i+1:03d}"] = ids
# split into parts at segment boundaries whose first shot is a fresh image (no cross-part "from")
n = len(shots); cuts = []
for p in range(1, a.parts):
    target = n * p // a.parts
    j = next(j for j in range(target, n) if "from" not in shots[j])
    cuts.append(j)
bounds = [0] + cuts + [n]
for p in range(a.parts):
    part = shots[bounds[p]:bounds[p + 1]]
    json.dump(part, open(P(f"qshots_part{p+1}.json"), "w"), indent=0)
    print(f"part{p+1}: {len(part)} images")
json.dump(picks, open(P("picks.json"), "w"), indent=0)
print("total", n, "images, avg hold", round(D["total"] / n, 2), "s")
