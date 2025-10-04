import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Fix import path - add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now we can import from src folder
try:
    from src.recommend import Recommender
    from src.embed_items import compute_text_embeddings, compute_image_embeddings

ROOT = os.path.dirname(__file__)
CSV_PATH = os.path.join(ROOT, 'products_preprocessed.csv')
EMB_PATH = os.path.join(ROOT, 'embeddings.npy')

# Check if required files exist
if not os.path.exists(CSV_PATH):
    st.error("❌ products_preprocessed.csv not found!")
    st.info("Please upload this file to your repository root or run preprocessing")
    st.stop()

if not os.path.exists(EMB_PATH):
    st.warning("⚠️ embeddings.npy not found!")
    st.info("This file will be generated when you first use the app")
    # You could add logic here to generate embeddings if needed

@st.cache_resource
def get_recommender():
    try:
        return Recommender(CSV_PATH, EMB_PATH)
    except Exception as e:
        st.error(f"❌ Error loading recommender: {str(e)}")
        st.stop()

st.set_page_config(page_title="ShopSmart - Recommendations", layout="wide")
st.title("🛍️ ShopSmart Recommendations")
st.caption("Find products you love using AI-powered similarity search.")

# Test if everything is working
try:
    rec = get_recommender()
    st.success("✅ Recommender loaded successfully!")
except:
    st.error("❌ Failed to load recommender system")
    st.stop()

tab1, tab2, tab3 = st.tabs(["By Product", "By Text", "By Image"]) 

with tab1:
    num = len(rec.products)
    if num == 0:
        st.warning("No products available. Run the pipeline first.")
    else:
        idx = st.number_input("Product index", min_value=0, max_value=num-1, value=0, step=1)
        k = st.slider("Top-K", 1, 20, 5)
        if st.button("Recommend Similar"):
            try:
                out = rec.recommend_by_index(int(idx), k)
                st.subheader("Query Product")
                qp = rec.products.iloc[int(idx)]
                colq1, colq2 = st.columns([1,2])
                with colq1:
                    if qp.get('image_path') and os.path.isfile(str(qp['image_path'])):
                        st.image(str(qp['image_path']), caption=qp.get('product_name', ''))
                    else:
                        st.write("📷 No image available")
                with colq2:
                    st.markdown(f"**{qp.get('product_name','')}**")
                    st.write(qp.get('description',''))
                st.divider()
                st.subheader("Similar Products")
                for _, row in out.iterrows():
                    c1, c2 = st.columns([1,2])
                    with c1:
                        if row.get('image_path') and os.path.isfile(str(row['image_path'])):
                            st.image(str(row['image_path']), use_column_width=True)
                        else:
                            st.write("📷 No image")
                    with c2:
                        st.markdown(f"**{row.get('product_name','')}**")
                        st.write(row.get('description',''))
                        st.caption(f"Score: {row.get('score', 0):.3f}")
            except Exception as e:
                st.error(f"❌ Error in recommendation: {str(e)}")

with tab2:
    q = st.text_input("Describe what you're looking for", "Elegant leather watch with brown strap")
    k = st.slider("Top-K (text)", 1, 20, 5, key='k_text')
    if st.button("Search by Text"):
        try:
            q_emb = compute_text_embeddings([q])
            zero_img = np.zeros((1, rec.embeddings.shape[1] - q_emb.shape[1]), dtype=np.float32)
            q_vec = np.concatenate([q_emb.astype(np.float32), zero_img], axis=1)
            sims = (rec.embeddings @ q_vec.T).flatten()
            top_idx = np.argsort(-sims)[:k]
            res = rec.products.iloc[top_idx].assign(score=sims[top_idx])
            for _, row in res.iterrows():
                c1, c2 = st.columns([1,2])
                with c1:
                    if row.get('image_path') and os.path.isfile(str(row['image_path'])):
                        st.image(str(row['image_path']), use_column_width=True)
                    else:
                        st.write("📷 No image")
                with c2:
                    st.markdown(f"**{row.get('product_name','')}**")
                    st.write(row.get('description',''))
                    st.caption(f"Score: {row.get('score', 0):.3f}")
        except Exception as e:
            st.error(f"❌ Error in text search: {str(e)}")

with tab3:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    k = st.slider("Top-K (image)", 1, 20, 5, key='k_image')
    if file is not None and st.button("Search by Image"):
        try:
            temp_path = os.path.join(ROOT, f"_streamlit_{file.name}")
            with open(temp_path, 'wb') as f:
                f.write(file.read())
            try:
                img_emb = compute_image_embeddings([temp_path])
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            zero_txt = np.zeros((1, rec.embeddings.shape[1] - img_emb.shape[1]), dtype=np.float32)
            q_vec = np.concatenate([zero_txt, img_emb.astype(np.float32)], axis=1)
            sims = (rec.embeddings @ q_vec.T).flatten()
            top_idx = np.argsort(-sims)[:k]
            res = rec.products.iloc[top_idx].assign(score=sims[top_idx])
            for _, row in res.iterrows():
                c1, c2 = st.columns([1,2])
                with c1:
                    if row.get('image_path') and os.path.isfile(str(row['image_path'])):
                        st.image(str(row['image_path']), use_column_width=True)
                    else:
                        st.write("📷 No image")
                with c2:
                    st.markdown(f"**{row.get('product_name','')}**")
                    st.write(row.get('description',''))
                    st.caption(f"Score: {row.get('score', 0):.3f}")
        except Exception as e:
            st.error(f"❌ Error in image search: {str(e)}")

