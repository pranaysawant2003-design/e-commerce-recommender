import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import pandas as pd
from .recommend import Recommender
from .embed_items import compute_text_embeddings, compute_image_embeddings
from .model import combine_embeddings


app = FastAPI(title="E-commerce Recommender API")

ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(ROOT, 'products_preprocessed.csv')
EMB_PATH = os.path.join(ROOT, 'embeddings.npy')


def get_recommender() -> Recommender:
    return Recommender(CSV_PATH, EMB_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{index}")
def recommend_index(index: int, k: int = 5):
    rec = get_recommender()
    df = rec.recommend_by_index(index, top_k=k)
    return df.to_dict(orient='records')


@app.post("/recommend/text")
async def recommend_text(query: str = Form(...), k: int = Form(5)):
    rec = get_recommender()
    q_text = query.strip()
    q_emb = compute_text_embeddings([q_text])
    # Build query vs existing embeddings by concatenating with zero image vector
    zero_img = np.zeros((1, rec.embeddings.shape[1] - q_emb.shape[1]), dtype=np.float32)
    q_combined = np.concatenate([q_emb.astype(np.float32), zero_img], axis=1)
    sims = (rec.embeddings @ q_combined.T).flatten()
    top_idx = np.argsort(-sims)[:k]
    return rec.products.iloc[top_idx].assign(score=sims[top_idx]).to_dict(orient='records')


@app.post("/recommend/image")
async def recommend_image(file: UploadFile = File(...), k: int = Form(5)):
    # Save to temp and compute embedding
    temp_path = os.path.join(ROOT, f"_upload_{file.filename}")
    with open(temp_path, 'wb') as f:
        f.write(await file.read())
    try:
        img_emb = compute_image_embeddings([temp_path])
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
    rec = get_recommender()
    # Build query vector with zero text
    zero_txt = np.zeros((1, rec.embeddings.shape[1] - img_emb.shape[1]), dtype=np.float32)
    q_combined = np.concatenate([zero_txt, img_emb.astype(np.float32)], axis=1)
    sims = (rec.embeddings @ q_combined.T).flatten()
    top_idx = np.argsort(-sims)[:k]
    return rec.products.iloc[top_idx].assign(score=sims[top_idx]).to_dict(orient='records')


