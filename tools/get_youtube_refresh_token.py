"""Run this ONCE on your own PC (not in the cloud) to get a YouTube refresh token.
  pip install google-auth-oauthlib
  python get_youtube_refresh_token.py path\\to\\client_secret.json
A browser opens -> sign in with the channel's Google account -> allow access.
Then copy the printed refresh token into the cloud environment's settings as YT_REFRESH_TOKEN."""
import sys
from google_auth_oauthlib.flow import InstalledAppFlow
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
flow = InstalledAppFlow.from_client_secrets_file(sys.argv[1] if len(sys.argv) > 1 else "client_secret.json", SCOPES)
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
print("\nYT_REFRESH_TOKEN =", creds.refresh_token)
print("(Keep this private. Paste it only into the environment settings, never into a chat.)")
