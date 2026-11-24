"""Calm, curious documentary bed: soft plucked arpeggios, warm pad, light clock-tick percussion.
Original synthesis (no samples). Length follows the voiceover.
usage: python pipeline/music_curious.py projects/<name> [--bpm 92] [--seed 7]"""
import argparse, json, os, wave
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("--bpm", type=float, default=92)
ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
rng = np.random.default_rng(a.seed)
SR = 44100
total = json.load(open(os.path.join(a.proj, "words.json")))["total"] + 4.0
N = int(SR * total); out = np.zeros((N, 2))
beat = 60.0 / a.bpm; bar = 4 * beat

def note(f, dur, amp, bright=1.0):
    t = np.arange(int(SR * dur)) / SR
    env = np.exp(-t * (3.2 / bright)) * np.minimum(1, t * 400)
    y = sum((0.6 ** k) * np.sin(2 * np.pi * f * (k + 1) * t + rng.uniform(0, 6)) for k in range(5))
    return amp * env * y

def pad(freqs, dur, amp):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, t / 1.2) * np.minimum(1, (dur - t) / 1.2)
    y = sum(np.sin(2 * np.pi * f * d * t) for f in freqs for d in (0.997, 1.003))
    return amp * env * y / (2 * len(freqs))

def tick(amp, hi=True):
    t = np.arange(int(SR * 0.03)) / SR
    noise = rng.uniform(-1, 1, len(t))
    return amp * noise * np.exp(-t * (260 if hi else 180)) * np.sin(2 * np.pi * (3200 if hi else 2400) * t)

def put(sig, at, pan=0.0):
    i = int(at * SR)
    if i >= N: return
    sig = sig[: N - i]
    out[i:i + len(sig), 0] += sig * (1 - pan) / 2 * 2 ** 0.5
    out[i:i + len(sig), 1] += sig * (1 + pan) / 2 * 2 ** 0.5

hz = lambda m: 440 * 2 ** ((m - 69) / 12)
# Am - F - C - G  /  Am - F - Dm - E   (gentle minor-to-major lift, "discovery" feel)
prog = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62], [57, 60, 64], [53, 57, 60], [50, 53, 57], [52, 56, 59]]
patt = [0, 1, 2, 1, 2, 1, 0, 2]   # 8th-note arpeggio pattern (chord tone index, +12 every other)
nbars = int(total / bar) + 1
for b in range(nbars):
    t0 = b * bar; ch = prog[b % len(prog)]
    section = (b // 8) % 4          # 0 intro-ish, 1 full, 2 lighter, 3 full
    put(pad([hz(m) for m in ch] + [hz(ch[0] - 12)], bar + 1.0, 0.16), t0)
    if section != 0 or b >= 4:
        for k in range(8):
            m = ch[patt[k]] + (12 if k % 2 else 24 if k == 7 else 12)
            put(note(hz(m), 1.2, 0.07 if section != 2 else 0.05, bright=0.8), t0 + k * beat / 2 + rng.normal(0, 0.004), pan=0.35 * np.sin(k))
    put(note(hz(ch[0] - 24), bar, 0.10, bright=2.5), t0)                         # soft bass root
    for k in range(8):                                                           # tick-tock on 8ths
        put(tick(0.10 if k % 2 == 0 else 0.06, hi=(k % 2 == 0)), t0 + k * beat / 2, pan=-0.2 if k % 2 else 0.2)
# fades
fi, fo = int(SR * 2.5), int(SR * 4.0)
out[:fi] *= np.linspace(0, 1, fi)[:, None]; out[-fo:] *= np.linspace(1, 0, fo)[:, None]
out /= np.abs(out).max() / 0.8
with wave.open(os.path.join(a.proj, "music.wav"), "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes())
print(f"music.wav {total:.1f}s @ {a.bpm} bpm")
