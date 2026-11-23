"""Original procedural synthwave bed (royalty-free): pads, bass, arpeggio, drums, with section changes.
usage: python pipeline/music_synthwave.py out.wav --seconds 420 [--bpm 96]"""
import argparse, wave
import numpy as np
SR = 44100
def note(n): return 440.0 * 2 ** ((n - 69) / 12)
def saw(f, t): return 2 * (t * f - np.floor(0.5 + t * f))
def lowpass(x, a):
    y = np.empty_like(x); acc = 0.0
    for i in range(0, len(x), 64):                      # cheap block one-pole
        blk = x[i:i + 64]; acc = acc + a * (blk.mean() - acc); y[i:i + 64] = acc + (blk - blk.mean()) * a
    return y
def env(n, a, d):
    e = np.ones(n); ai = int(a * SR); e[:ai] = np.linspace(0, 1, ai) if ai else 1
    return e * np.exp(-np.arange(n) / (d * SR))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seconds", type=float, default=420); ap.add_argument("--bpm", type=float, default=96)
    a = ap.parse_args()
    N = int(SR * a.seconds) + SR * 4
    mix = np.zeros(N)
    beat = 60 / a.bpm; bar = beat * 4
    prog = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [48, 52, 55]), (55, [55, 59, 62])]   # Am F C G
    nbars = int(a.seconds / bar) + 1
    rng = np.random.default_rng(7)
    for b in range(nbars):
        t0 = b * bar; s0 = int(t0 * SR); n = int(bar * SR)
        root, chord = prog[b % 4]
        section = (b // 8) % 4                           # 0 intro-ish, 1 full, 2 breakdown, 3 full+
        tt = np.arange(n) / SR
        pad = sum(saw(note(c), tt) + 0.6 * saw(note(c) * 1.006, tt) for c in chord) / 6
        pad *= np.minimum(1, np.minimum(tt / 0.4, (bar - tt) / 0.3)) * 0.22
        mix[s0:s0 + n] += pad
        if section != 2:                                  # bass on 8ths
            for k in range(8):
                ss = s0 + int(k * beat / 2 * SR); m = int(beat / 2 * SR)
                tb = np.arange(m) / SR
                mix[ss:ss + m] += np.sign(np.sin(2 * np.pi * note(root - 12) * tb)) * env(m, 0.005, 0.12) * 0.16
        if section in (1, 3) or (section == 0 and b % 8 >= 4):  # arpeggio 16ths
            arp = chord + [chord[1] + 12]
            for k in range(16):
                ss = s0 + int(k * beat / 4 * SR); m = int(beat / 4 * SR * 1.5)
                ta = np.arange(m) / SR
                mix[ss:ss + m] += saw(note(arp[k % 4] + 12), ta) * env(m, 0.002, 0.09) * 0.09
        if section != 2:                                  # drums
            for k in range(4):
                ss = s0 + int(k * beat * SR); m = int(0.25 * SR); tk = np.arange(m) / SR
                kick = np.sin(2 * np.pi * np.cumsum(50 + 120 * np.exp(-tk * 30)) / SR) * np.exp(-tk * 12)
                mix[ss:ss + m] += kick * 0.5
                mix[ss:ss + int(beat * SR)] *= np.minimum(1, 0.55 + np.arange(int(beat * SR)) / (0.18 * SR))[:len(mix[ss:ss + int(beat * SR)])]  # pump
                if k in (1, 3):
                    sn = rng.uniform(-1, 1, int(0.18 * SR)) * np.exp(-np.arange(int(0.18 * SR)) / (0.05 * SR))
                    mix[ss:ss + len(sn)] += sn * 0.18
                for h in (0, 0.5):
                    hs = ss + int(h * beat * SR); hn = rng.uniform(-1, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / (0.008 * SR))
                    mix[hs:hs + len(hn)] += np.diff(np.concatenate([[0], hn])) * 0.15
    mix = mix[:int(SR * a.seconds)]
    f = int(SR * 3); mix[-f:] *= np.linspace(1, 0, f); mix[:SR] *= np.linspace(0, 1, SR)
    mix /= np.max(np.abs(mix)) * 1.1
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
    print("wrote", a.out, a.seconds, "s")
main()
