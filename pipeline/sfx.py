"""Original synthesized sound effects (no samples, free to use): writes assets/sfx/<name>.wav (48 kHz stereo).
hit = cinematic impact (sub boom + crack), riser = noise sweep up, whoosh = air swish, pop = UI pop,
ding = bell, tick = clock tick, swoosh_hit = whoosh into impact (the attention-grabber for the first second).
usage: python pipeline/sfx.py"""
import os, wave
import numpy as np
SR = 48000
rng = np.random.default_rng(3)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sfx")

def tt(d): return np.arange(int(SR * d)) / SR
def lp(x, fc):   # one-pole low-pass, fc may be an array
    a = np.exp(-2 * np.pi * np.broadcast_to(fc, x.shape) / SR); y = np.empty_like(x); s = 0.0
    for i in range(len(x)): s = (1 - a[i]) * x[i] + a[i] * s; y[i] = s
    return y
def bp(x, fc, q=1.5): return lp(x, fc * (1 + 1 / q)) - lp(x, fc / (1 + 1 / q))
def norm(x, peak=0.9): return x / (np.abs(x).max() + 1e-9) * peak

def hit():
    t = tt(1.6)
    f = 38 + 50 * np.exp(-t * 9)                              # pitch-dropping sub
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.6)
    crack = lp(rng.uniform(-1, 1, len(t)), 2500) * np.exp(-t * 28)
    body = bp(rng.uniform(-1, 1, len(t)), 180) * np.exp(-t * 6) * 3
    return norm(1.0 * sub + 0.55 * crack + 0.5 * body)
def riser(d=1.2):
    t = tt(d); k = t / d
    n = bp(rng.uniform(-1, 1, len(t)), 300 + 5000 * k ** 2, 2.5)
    tone = np.sin(2 * np.pi * np.cumsum(200 + 700 * k ** 2) / SR) * 0.25
    return norm((n + tone) * k ** 2.2)
def whoosh(d=0.55):
    t = tt(d); k = t / d; env = np.sin(np.pi * k) ** 2
    return norm(bp(rng.uniform(-1, 1, len(t)), 400 + 2600 * np.sin(np.pi * k), 1.2) * env)
def pop():
    t = tt(0.12); f = 900 * np.exp(-t * 22) + 260
    return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 30) * np.minimum(1, t * 2000), 0.8)
def ding():
    t = tt(1.8); y = sum(a * np.sin(2 * np.pi * 1320 * r * t) * np.exp(-t * dcy) for r, a, dcy in
                        [(1, 1, 2.2), (2.76, 0.5, 4), (5.4, 0.25, 7), (0.5, 0.3, 1.5)])
    return norm(y * np.minimum(1, t * 3000), 0.7)
def tick():
    t = tt(0.05); return norm(bp(rng.uniform(-1, 1, len(t)), 3000, 3) * np.exp(-t * 160), 0.7)
def swoosh_hit():
    w = whoosh(0.45); h = hit(); y = np.zeros(int(SR * 0.38) + len(h)); y[:len(w)] += 0.6 * w; y[int(SR * 0.38):] += h
    return norm(y)

def write(name, y, width=0.15):
    os.makedirs(OUT, exist_ok=True)
    st = np.stack([y, np.roll(y, int(SR * 0.0004 * width * 10))], 1)    # tiny stereo spread
    with wave.open(os.path.join(OUT, name + ".wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    for n, f in [("hit", hit), ("riser", riser), ("whoosh", whoosh), ("pop", pop), ("ding", ding), ("tick", tick), ("swoosh_hit", swoosh_hit)]:
        write(n, f()); print(n)
