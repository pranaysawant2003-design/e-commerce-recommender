import os
import numpy as np
from .dataset import load_products
from .embed_items import compute_text_embeddings, compute_image_embeddings
from .model import combine_embeddings


def build_embeddings(data_csv: str, output_path: str) -> str:
    df = load_products(data_csv)
    text_emb = compute_text_embeddings(df['text'].tolist())
    image_emb = compute_image_embeddings(df['image_path'].tolist())
    combined = combine_embeddings(text_emb, image_emb)
    np.save(output_path, combined)
    return output_path


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(__file__))
    data_csv = os.path.join(root, 'products_preprocessed.csv')
    out_npy = os.path.join(root, 'embeddings.npy')
    print(f"Building embeddings from {data_csv}")
    path = build_embeddings(data_csv, out_npy)
    print(f"Saved embeddings: {path}")


