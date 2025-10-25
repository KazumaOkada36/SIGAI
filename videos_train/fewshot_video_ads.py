#!/usr/bin/env python3
import os, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Video decoding: prefer decord, fallback to OpenCV
_HAS_DECORD = False
try:
    import decord
    decord.bridge.set_bridge('torch')
    _HAS_DECORD = True
except Exception:
    import cv2

import torch
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score

def load_clip(name, pretrained, device):
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(name)
    return model.to(device), preprocess, tokenizer

def sample_frames_decord(video_path, num_frames):
    vr = decord.VideoReader(video_path)
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty video: {video_path}")
    if n <= num_frames:
        idxs = list(range(n))
        while len(idxs) < num_frames:
            idxs += idxs[:max(0, num_frames - len(idxs))]
        idxs = idxs[:num_frames]
    else:
        step = n / num_frames
        idxs = [int(i*step + step/2) for i in range(num_frames)]
    batch = vr.get_batch(idxs)   # (T,H,W,3)
    frames = [batch[i].asnumpy() for i in range(batch.shape[0])]
    return frames

def sample_frames_cv2(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        idxs = list(range(num_frames))
    elif total <= num_frames:
        idxs = list(range(total))
        while len(idxs) < num_frames:
            idxs += idxs[:max(0, num_frames - len(idxs))]
        idxs = idxs[:num_frames]
    else:
        step = total / num_frames
        idxs = [int(i*step + step/2) for i in range(num_frames)]
    frames = []
    for target in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
    if not frames:
        raise RuntimeError(f"No frames read: {video_path}")
    return frames[:num_frames]

def encode_video_clip(model, preprocess, device, frames):
    from PIL import Image
    imgs = []
    for arr in frames:
        img = Image.fromarray(arr)
        imgs.append(preprocess(img).unsqueeze(0).to(device))
    batch = torch.cat(imgs, dim=0)  # [T,3,224,224]
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.mean(dim=0).cpu().numpy()  # [D]

def zero_shot_probs(model, tokenizer, device, prompts, vid_feats):
    with torch.no_grad():
        tokens = tokenizer(prompts)
        tfeat = model.encode_text(tokens.to(device))
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        v = torch.tensor(vid_feats, device=device)
        logits = v @ tfeat.T
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

def fit_loocv_logreg(X, y_str):
    classes = sorted(list(set(y_str)))
    cls2id = {c:i for i,c in enumerate(classes)}
    y = np.array([cls2id[s] for s in y_str])
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    probmat = np.zeros((len(y), len(classes)))
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[test_idx])
        probmat[test_idx] = probs
        preds[test_idx] = probs.argmax(axis=1)
    macro = f1_score(y, preds, average="macro")
    return macro, preds, probmat, cls2id, classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=12)
    ap.add_argument("--clip_model", default="ViT-L-14")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--products", type=str, default="beverage,restaurant,app,automobile,financial service,fashion")
    ap.add_argument("--tones", type=str, default="humorous,inspirational,urgent,sentimental")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    if not {"video_id","path"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: video_id,path")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_clip(args.clip_model, args.clip_pretrained, device)

    feats, kept = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding videos"):
        path = str(row["path"])
        if not os.path.exists(path):
            print(f"[WARN] missing file: {path}")
            continue
        try:
            frames = sample_frames_decord(path, args.num_frames) if _HAS_DECORD else sample_frames_cv2(path, args.num_frames)
            vfeat = encode_video_clip(model, preprocess, device, frames)
            feats.append(vfeat); kept.append({"video_id": row["video_id"], "path": path})
        except Exception as e:
            print(f"[WARN] failed {path}: {e}")
            continue

    X = np.stack(feats, axis=0)
    with open(os.path.join(args.out_dir,"index.json"),"w") as f:
        json.dump(kept, f, indent=2)
    np.save(os.path.join(args.out_dir,"embeddings.npy"), X)

    # Zero-shot
    products = [s.strip() for s in args.products.split(",") if s.strip()]
    tones = [s.strip() for s in args.tones.split(",") if s.strip()]
    prod_prompts = [f"a {c} advertisement" for c in products]
    tone_prompts = [f"a {c} advertisement" for c in tones]
    prod_probs = zero_shot_probs(model, tokenizer, device, prod_prompts, X)
    tone_probs = zero_shot_probs(model, tokenizer, device, tone_prompts, X)

    zs_rows = []
    for i, meta in enumerate(kept):
        row = {"video_id": meta["video_id"], "path": meta["path"],
               "zs_product_top1": products[int(prod_probs[i].argmax())],
               "zs_tone_top1": tones[int(tone_probs[i].argmax())]}
        for j, c in enumerate(products): row[f"zs_product_prob[{c}]"] = float(prod_probs[i,j])
        for j, c in enumerate(tones):    row[f"zs_tone_prob[{c}]"]    = float(tone_probs[i,j])
        zs_rows.append(row)
    pd.DataFrame(zs_rows).to_csv(os.path.join(args.out_dir,"zero_shot.csv"), index=False)

    # LOOCV linear probe if labels are present
    report = []
    if "product_label" in df.columns and "tone_label" in df.columns:
        dfk = pd.DataFrame(kept).merge(df[["video_id","product_label","tone_label"]], on="video_id", how="left")
        out = {"video_id": dfk["video_id"].tolist()}

        if dfk["product_label"].nunique(dropna=True) >= 2:
            macro, preds, probmat, _, classes = fit_loocv_logreg(X, dfk["product_label"].fillna("").tolist())
            report.append(f"LOOCV product macro-F1: {macro:.3f} ({len(classes)} classes)")
            out["loocv_product_pred"] = [classes[p] for p in preds]
            for j,c in enumerate(classes): out[f"loocv_product_prob[{c}]"] = probmat[:,j].tolist()
        else:
            report.append("LOOCV product: skipped (need ≥2 classes)")

        if dfk["tone_label"].nunique(dropna=True) >= 2:
            macro, preds, probmat, _, classes = fit_loocv_logreg(X, dfk["tone_label"].fillna("").tolist())
            report.append(f"LOOCV tone macro-F1: {macro:.3f} ({len(classes)} classes)")
            out["loocv_tone_pred"] = [classes[p] for p in preds]
            for j,c in enumerate(classes): out[f"loocv_tone_prob[{c}]"] = probmat[:,j].tolist()
        else:
            report.append("LOOCV tone: skipped (need ≥2 classes)")

        cv_df = pd.DataFrame(out)
        zs_df = pd.DataFrame(zs_rows)
        merged = pd.merge(zs_df, cv_df, on="video_id", how="left")
        merged.to_csv(os.path.join(args.out_dir,"cv_predictions.csv"), index=False)

        with open(os.path.join(args.out_dir,"report.txt"),"w") as f:
            f.write("\n".join(report))

    print("Done. Saved to", args.out_dir)

if __name__ == "__main__":
    main()
