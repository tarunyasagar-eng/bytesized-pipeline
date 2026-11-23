"""Run the Kaggle CLI with a proxy-injected token (skips local token introspection)."""
import os, sys
os.environ.setdefault("KAGGLE_API_TOKEN", "proxy-injected")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
from kaggle.api.kaggle_api_extended import KaggleApi
KaggleApi._introspect_token = lambda self, tok: os.environ.get("KAGGLE_USERNAME", "tarunyasagar")
from kaggle.cli import main
sys.exit(main())
