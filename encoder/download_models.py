"""One-time download of jina-reranker-v3 into the ``ikincitik_models`` volume.

The hub is online only for this step:
    docker compose run --rm -e HF_HUB_OFFLINE=0 --no-deps encoder python download_models.py

The snapshot lands in HF cache layout under /models/hub, so the encoder loads it
by its snapshot path, and pinning the revision also pins the remote model code
(modeling.py) that trust_remote_code runs.
"""

import os

from huggingface_hub import snapshot_download

REPO = "jinaai/jina-reranker-v3"
REVISION = os.environ.get("JINA_RERANKER_REVISION", "d7d7e73b6ea138ced340b83865931b5dfb6c97aa")

path = snapshot_download(REPO, revision=REVISION, cache_dir="/models/hub")
print(path)
for f in sorted(os.listdir(path)):
    print(" ", f)
