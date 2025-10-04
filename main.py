import os
from .preprocess import merge_all_csvs
from .train import build_embeddings
from .recommend import Recommender


def run_pipeline(data_dir: str, merged_csv: str, embeddings_path: str):
    if not os.path.isfile(merged_csv):
        merge_all_csvs(data_dir, merged_csv)
    build_embeddings(merged_csv, embeddings_path)


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, 'data')
    merged_csv = os.path.join(root, 'products_preprocessed.csv')
    embeddings_path = os.path.join(root, 'embeddings.npy')

    print("Running full pipeline...")
    run_pipeline(data_dir, merged_csv, embeddings_path)

    rec = Recommender(merged_csv, embeddings_path)
    # Demo: show top 5 similar to index 0 if exists
    if len(rec.products) > 0:
        result = rec.recommend_by_index(0, top_k=5)
        print("Query:")
        print(rec.products.iloc[0][['product_name', 'description', 'image_path']])
        print("Top-5 similar:")
        print(result[['product_name', 'description', 'image_path', 'score']])
    else:
        print("No products found.")


