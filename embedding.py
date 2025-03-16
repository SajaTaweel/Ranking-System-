import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from logger import logger

# Load Pretrained Transformer Model & Tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract CLS token
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return np.zeros((model.config.hidden_size,))
