import numpy as np


def combine_embeddings(text_embeds: np.ndarray, image_embeds: np.ndarray) -> np.ndarray:
    if text_embeds.shape[0] != image_embeds.shape[0]:
        raise ValueError("Text and image embeddings must have the same number of rows")
    # L2-normalize each modality, then concatenate
    def l2norm(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / denom

    t = l2norm(text_embeds)
    i = l2norm(image_embeds)
    combined = np.concatenate([t, i], axis=1).astype(np.float32)
    return combined


