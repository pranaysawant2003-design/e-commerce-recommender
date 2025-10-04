import os
from typing import Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms


def compute_text_embeddings(texts, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(texts), batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def _load_resnet50() -> Tuple[torch.nn.Module, transforms.Compose]:
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return resnet, preprocess


def compute_image_embeddings(image_paths, device: str | None = None) -> np.ndarray:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = _load_resnet50()
    model = model.to(device)

    features = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc='Image embeddings'):
            try:
                if not path or not os.path.isfile(path):
                    features.append(np.zeros(2048, dtype=np.float32))
                    continue
                img = Image.open(path).convert('RGB')
                tensor = preprocess(img).unsqueeze(0).to(device)
                vec = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
                features.append(vec)
            except Exception:
                features.append(np.zeros(2048, dtype=np.float32))
    return np.vstack(features)


