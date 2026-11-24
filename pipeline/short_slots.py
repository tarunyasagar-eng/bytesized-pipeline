"""Pick varied daily publish times for Shorts: one per day, a different time each day, always >= 12 h after the previous one.
Times are drawn from a 12-hour UTC window (10:00-21:59), so any two consecutive days are automatically 12-34 h apart.
usage: python pipeline/short_slots.py N [--start 2026-12-07]   (default start: the day after the last scheduled Short)"""
import argparse, datetime as dt, glob, json, random
WIN = (10 * 60, 22 * 60 - 1)   # minutes after 00:00 UTC
def last_scheduled():
    ts = [json.load(open(f)).get("publish_at") for f in glob.glob("projects/*/uploaded.json")]
    ts = [dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in ts if t]
    return max(ts) if ts else None
def slots(n, start=None):
    prev = last_scheduled()
    day = start or ((prev + dt.timedelta(days=1)).date() if prev else dt.date.today() + dt.timedelta(days=1))
    out = []
    for _ in range(n):
        while True:
            m = random.randint(*WIN)
            t = dt.datetime.combine(day, dt.time(m // 60, m % 60), dt.timezone.utc)
            if prev is None or (t - prev) >= dt.timedelta(hours=12): break
        out.append(t.strftime("%Y-%m-%dT%H:%M:00Z")); prev = t; day += dt.timedelta(days=1)
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("n", type=int); ap.add_argument("--start"); a = ap.parse_args()
    print("\n".join(slots(a.n, dt.date.fromisoformat(a.start) if a.start else None)))
