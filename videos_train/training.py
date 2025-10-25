import pandas as pd
import numpy as np

import pathlib

# Point this to your folder of videos
ROOT = "SIGAI/videos_train"
VID_DIR = f"{ROOT}"   # e.g., documents/APPLOVIN_CAL_HACKS/videos

# Add any extensions you use
EXTS = {".mp4", ".mov", ".mkv", ".avi"}

paths = []
for p in pathlib.Path(VID_DIR).rglob("*"):
    if p.is_file() and p.suffix.lower() in EXTS:
        paths.append(p.resolve())

# Build manifest (no labels)
df = pd.DataFrame({
    "video_id": [p.stem for p in paths],
    "path": [str(p) for p in paths],
})

df = df.sort_values("video_id").reset_index(drop=True)
df.to_csv(f"{ROOT}/videos_manifest.csv", index=False)

print(f"Found {len(df)} videos")
print(df.head())

