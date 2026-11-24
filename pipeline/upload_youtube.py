"""Upload a video to YouTube (Data API v3, resumable) and optionally set a custom thumbnail.
Needs env vars YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN (from your existing OAuth token.json).
Defaults to PRIVATE so you review before anything goes public.
usage: python pipeline/upload_youtube.py video.mp4 --meta projects/<name>/upload.json [--thumb thumb.jpg] [--publish-at 2026-11-25T13:30:00Z]"""
import argparse, json, os, sys, time, requests
CA = "/root/.ccr/ca-bundle.crt" if os.path.exists("/root/.ccr/ca-bundle.crt") else True

def access_token():
    miss = [k for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN") if not os.environ.get(k)]
    if miss:
        sys.exit(f"Missing environment variables: {', '.join(miss)}")
    r = requests.post("https://oauth2.googleapis.com/token", verify=CA, data=dict(
        client_id=os.environ["YT_CLIENT_ID"], client_secret=os.environ["YT_CLIENT_SECRET"],
        refresh_token=os.environ["YT_REFRESH_TOKEN"], grant_type="refresh_token"))
    r.raise_for_status()
    return r.json()["access_token"]

def upload(path, meta, publish_at=None):
    tok = access_token()
    status = {"privacyStatus": "private", "selfDeclaredMadeForKids": False}
    if publish_at:
        status["publishAt"] = publish_at          # stays private until this UTC time, then goes public
    body = {"snippet": {"title": meta["title"][:100], "description": meta.get("description", ""),
                        "tags": meta.get("tags", []), "categoryId": str(meta.get("categoryId", 27))},
            "status": status}
    size = os.path.getsize(path)
    init = requests.post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                         verify=CA, json=body, headers={"Authorization": f"Bearer {tok}", "X-Upload-Content-Type": "video/mp4",
                                                         "X-Upload-Content-Length": str(size)})
    init.raise_for_status()
    url = init.headers["Location"]
    sent = 0; chunk = 8 * 1024 * 1024
    with open(path, "rb") as f:
        while sent < size:
            f.seek(sent); data = f.read(chunk)
            for attempt in range(5):
                try:
                    r = requests.put(url, data=data, verify=CA, headers={"Authorization": f"Bearer {tok}",
                                     "Content-Range": f"bytes {sent}-{sent+len(data)-1}/{size}"}, timeout=300)
                    break
                except requests.RequestException:
                    time.sleep(2 ** attempt)
            if r.status_code in (200, 201):
                return r.json()["id"]
            if r.status_code == 308:
                rng = r.headers.get("Range"); sent = int(rng.split("-")[1]) + 1 if rng else sent
                print(f"uploaded {sent*100//size}%", flush=True)
            else:
                sys.exit(f"upload failed {r.status_code}: {r.text[:500]}")

def set_thumb(video_id, thumb):
    tok = access_token()
    r = requests.post(f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={video_id}", verify=CA,
                      data=open(thumb, "rb").read(), headers={"Authorization": f"Bearer {tok}", "Content-Type": "image/jpeg"})
    print("thumbnail:", r.status_code, r.text[:200])

def schedule(video_id, publish_at):
    """Set publishAt on an already-uploaded private video (it goes public automatically at that UTC time)."""
    tok = access_token()
    r = requests.put("https://www.googleapis.com/youtube/v3/videos?part=status", verify=CA,
                     headers={"Authorization": f"Bearer {tok}"},
                     json={"id": video_id, "status": {"privacyStatus": "private", "publishAt": publish_at, "selfDeclaredMadeForKids": False}})
    print("schedule:", video_id, publish_at, r.status_code, r.text[:200] if r.status_code != 200 else "ok")
    r.raise_for_status()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":     # python pipeline/upload_youtube.py --schedule VIDEO_ID 2026-12-01T14:00:00Z
        schedule(sys.argv[2], sys.argv[3]); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--meta", required=True); ap.add_argument("--thumb"); ap.add_argument("--publish-at")
    a = ap.parse_args()
    vid = upload(a.video, json.load(open(a.meta)), a.publish_at)
    print("VIDEO ID:", vid, f"https://youtu.be/{vid}  (private — review in YouTube Studio)")
    if a.thumb:
        set_thumb(vid, a.thumb)
