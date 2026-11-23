import asyncio, json, os, subprocess, wave, sys
import numpy as np, edge_tts
VOICE = sys.argv[1] if len(sys.argv) > 1 else "en-US-AndrewMultilingualNeural"
RATE, PITCH = "+9%", "+2Hz"
SR = 48000
SEG = [  # (text, pause after, scene line)
 ("In 1518, a woman in Strasbourg started dancing in the street...", 0.22, 0),
 ("and she couldn't stop.", 0.45, 0),
 ("She danced for days.", 0.22, 1),
 ("Her feet bled.", 0.18, 1),
 ("She kept going!", 0.42, 1),
 ("Within a month, hundreds of people had joined her!", 0.42, 2),
 ("So the city's doctors prescribed...", 0.55, 3),
 ("more dancing!", 0.40, 3),
 ("They built a stage, and hired musicians!", 0.42, 4),
 ("It only got worse.", 0.30, 5),
 ("Some accounts say dancers collapsed... and died.", 0.50, 5),
 ("Historians now think it was mass hysteria, sparked by famine and fear.", 0.42, 6),
 ("It's called the Dancing Plague.", 0.28, 7),
 ("And it wasn't the only one.", 0.40, 7),
]
async def synth(text):
    c = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH, boundary="WordBoundary",
                             proxy=os.environ.get("HTTPS_PROXY"))
    audio = b""; words = []
    async for ch in c.stream():
        if ch["type"] == "audio": audio += ch["data"]
        elif ch["type"] == "WordBoundary":
            words.append({"t": ch["offset"] / 1e7, "d": ch["duration"] / 1e7, "w": ch["text"]})
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                         input=audio, capture_output=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32), words
async def main():
    out = [np.zeros(int(SR * 0.05), np.float32)]; t = 0.05
    allw = []; line_t = {}
    for text, pause, li in SEG:
        a, ws = await synth(text)
        st = max(0.0, ws[0]["t"] - 0.04); en = min(len(a) / SR, ws[-1]["t"] + ws[-1]["d"] + 0.12)
        clip = a[int(st * SR):int(en * SR)]
        fade = int(SR * 0.01); clip[:fade] *= np.linspace(0, 1, fade); clip[-fade:] *= np.linspace(1, 0, fade)
        line_t.setdefault(li, t)
        for w in ws:
            allw.append({"t": round(t + w["t"] - st, 3), "d": round(w["d"], 3), "w": w["w"]})
        out += [clip, np.zeros(int(SR * pause), np.float32)]
        t += len(clip) / SR + pause
    y = np.concatenate(out)
    with wave.open("vo.wav", "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(y.astype(np.int16).tobytes())
    json.dump({"lines": [s[0] for s in SEG], "line_t": [line_t[i] for i in range(8)], "words": allw, "total": t},
              open("words.json", "w"), indent=0)
    print("duration", round(t, 2), "line starts", [round(line_t[i], 2) for i in range(8)])
asyncio.run(main())
