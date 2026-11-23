"""Voiceover from projects/<name>/script.json: one TTS clip per segment, joined with exact pauses.
Writes vo.wav + words.json (word timings, segment start times).
usage: python pipeline/tts.py projects/<name> [--voice en-US-BrianNeural]"""
import argparse, asyncio, json, os, subprocess, wave
import numpy as np, edge_tts
SR = 48000
async def synth(text, voice, rate, pitch):
    c = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary", proxy=os.environ.get("HTTPS_PROXY"))
    audio = b""; words = []
    async for ch in c.stream():
        if ch["type"] == "audio": audio += ch["data"]
        elif ch["type"] == "WordBoundary":
            words.append({"t": ch["offset"] / 1e7, "d": ch["duration"] / 1e7, "w": ch["text"]})
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                         input=audio, capture_output=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32), words
async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project"); ap.add_argument("--voice")
    ap.add_argument("--rate", default="+9%"); ap.add_argument("--pitch", default="+2Hz")
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.project, "script.json")))
    voice = a.voice or S.get("voice", "en-US-BrianNeural")
    out = [np.zeros(int(SR * 0.05), np.float32)]; t = 0.05; allw = []; seg_t = []
    for i, seg in enumerate(S["segments"]):
        for attempt in range(3):
            try:
                y, ws = await synth(seg["text"], voice, a.rate, a.pitch); break
            except Exception as e:
                print("retry", i, e); await asyncio.sleep(2)
        st = max(0.0, ws[0]["t"] - 0.04); en = min(len(y) / SR, ws[-1]["t"] + ws[-1]["d"] + 0.12)
        clip = y[int(st * SR):int(en * SR)].copy()
        f = int(SR * 0.01); clip[:f] *= np.linspace(0, 1, f); clip[-f:] *= np.linspace(1, 0, f)
        seg_t.append(round(t, 3))
        allw += [{"t": round(t + w["t"] - st, 3), "d": round(w["d"], 3), "w": w["w"], "seg": i} for w in ws]
        pause = seg.get("pause", 0.3)
        out += [clip, np.zeros(int(SR * pause), np.float32)]; t += len(clip) / SR + pause
    with wave.open(os.path.join(a.project, "vo.wav"), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(np.concatenate(out).astype(np.int16).tobytes())
    json.dump({"seg_t": seg_t, "words": allw, "total": t}, open(os.path.join(a.project, "words.json"), "w"))
    print(f"{len(S['segments'])} segments, {len(allw)} words, {t/60:.2f} min")
asyncio.run(main())
