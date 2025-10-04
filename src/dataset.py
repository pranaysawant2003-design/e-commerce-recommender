import os
import pandas as pd


def load_products(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ['product_name', 'description', 'image_path', 'category']:
        if col not in df.columns:
            df[col] = ''
    df['product_name'] = df['product_name'].fillna('')
    df['description'] = df['description'].fillna('')
    df['category'] = df['category'].fillna('')
    df['image_path'] = df['image_path'].fillna('')
    df['text'] = (df['product_name'].astype(str) + ' ' + df['description'].astype(str)).str.strip()
    return df


