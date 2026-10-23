#!/bin/bash
set -e

mkdir -p ~/TBI/Data/rsna_raw
sudo chown -R $USER:$USER ~/TBI/Data/rsna_raw

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "BŁĄD: Brak pliku ~/.kaggle/kaggle.json! Skonfiguruj dostęp do API Kaggle."
    exit 1
fi

if [ ! -f ~/TBI/Data/rsna_raw/stage_2_train.zip ] || [ ! -f ~/TBI/Data/rsna_raw/stage_2_train.csv ]; then
    echo "RSNA files missing. Starting download."
    URL=$(python3 -c '
import json, os, requests
with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
    c = json.load(f)
r = requests.get("https://www.kaggle.com/api/v1/competitions/data/download-all/rsna-intracranial-hemorrhage-detection", auth=(c["username"], c["key"]), allow_redirects=False)
print(r.headers["Location"])
    ')
    aria2c -x 16 -s 16 -c -d ~/TBI/Data/rsna_raw -o stage_2_train.zip "$URL"
    unzip -o -j ~/TBI/Data/rsna_raw/stage_2_train.zip "*stage_2_train.csv" -d ~/TBI/Data/rsna_raw/
else
    echo "RSNA files already exist. Skipping download."
fi

DOCKER_BUILDKIT=0 docker-compose build --no-cache tbi_app && \
docker-compose up -d tbi_postgres tbi_mongo tbi_redis tbi-minio && \
sleep 5 && \
docker-compose run --rm tbi_app python -u App/RunAll.py