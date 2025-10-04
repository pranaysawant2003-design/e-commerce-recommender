import os
import re
import pandas as pd
from typing import List


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {
        'title': 'product_name',
        'name': 'product_name',
        'product_title': 'product_name',
        'product_name': 'product_name',
        'desc': 'description',
        'product_description': 'description',
        'description': 'description',
        'category': 'category',
        'categories': 'category',
        'img': 'image_path',
        'image': 'image_path',
        'image_link': 'image_path',
        'image_url': 'image_path',
        'image_path': 'image_path',
    }

    df = df.rename(columns={c: column_mapping.get(c, c) for c in df.columns})
    for col in ['product_name', 'description', 'image_path', 'category']:
        if col not in df.columns:
            df[col] = ""
    df['product_name'] = df['product_name'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)
    df['category'] = df['category'].apply(clean_text)
    return df[['product_name', 'description', 'image_path', 'category']]


def preprocess_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = normalize_columns(df)
    df = df.dropna(subset=['product_name']).reset_index(drop=True)
    return df


def merge_all_csvs(data_dir: str, output_csv: str) -> None:
    csv_files: List[str] = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.lower().endswith('.csv')
    ]
    frames: List[pd.DataFrame] = []
    for csv_f in csv_files:
        try:
            frames.append(preprocess_csv(csv_f))
        except Exception as exc:
            # Skip malformed files but continue processing others
            print(f"[WARN] Skipping {csv_f}: {exc}")
    if not frames:
        raise RuntimeError("No valid CSV files found to preprocess.")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['product_name', 'description']).reset_index(drop=True)
    merged.to_csv(output_csv, index=False)


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    output_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'products_preprocessed.csv')
    print(f"Preprocessing CSVs from: {data_dir}")
    merge_all_csvs(data_dir, output_csv)
    print(f"Saved: {output_csv}")


