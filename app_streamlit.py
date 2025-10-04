import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Add this at the very beginning to test
st.write("🧪 Testing - If you see this, Streamlit is working!")


# Fix import path for relative imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import your modules with absolute imports
from src.recommend import Recommender
from src.embed_items import compute_text_embeddings, compute_image_embeddings

ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(ROOT, 'products_preprocessed.csv')
EMB_PATH = os.path.join(ROOT, 'embeddings.npy')

@st.cache_resource
def get_recommender():
    return Recommender(CSV_PATH, EMB_PATH)

st.set_page_config(page_title="ShopSmart - Recommendations", layout="wide")
st.title("🛍️ ShopSmart Recommendations")
st.caption("Find products you love using AI-powered similarity search.")

tab1, tab2, tab3 = st.tabs(["By Product", "By Text", "By Image"]) 

with tab1:
    rec = get_recommender()
    num = len(rec.products)
    if num == 0:
        st.warning("No products available. Run the pipeline first.")
    else:
        idx = st.number_input("Product index", min_value=0, max_value=num-1, value=0, step=1)
        k = st.slider("Top-K", 1, 20, 5)
        if st.button("Recommend Similar"):
            out = rec.recommend_by_index(int(idx), k)
            st.subheader("Query Product")
            qp = rec.products.iloc[int(idx)]
            colq1, colq2 = st.columns([1,2])
            with colq1:
                if qp.get('image_path') and os.path.isfile(str(qp['image_path'])):
                    st.image(str(qp['image_path']), caption=qp.get('product_name', ''))
            with colq2:
                st.markdown(f"**{qp.get('product_name','')}**")
                st.write(qp.get('description',''))
            st.divider()
            st.subheader("Similar Products")
            for _, row in out.iterrows():
                c1, c2 = st.columns([1,2])
                with c:
