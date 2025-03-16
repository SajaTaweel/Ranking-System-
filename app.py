from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import yaml
from logger import logger
from embedding import get_embedding  # Import the get_embedding function to generate embeddings

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize FastAPI app
app = FastAPI()


# Load the trained model
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_dim,
            config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            batch_first=True
        )
        self.fc = torch.nn.Linear(config["model"]["hidden_dim"], 2)  # Binary classification

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        return self.fc(last_hidden_state)


# Load the model weights
model = LSTMClassifier(config["model"]["input_dim"])
model.load_state_dict(torch.load(config["model_path"]))
model.eval()

# Initialize tokenizer (for embedding generation)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Pydantic model for incoming request data
class TextInput(BaseModel):
    text: str


# Define prediction endpoint
@app.post("/predict")
async def predict_sentiment(input: TextInput):
    try:
        # Step 1: Clean and preprocess the input text
        embedding = get_embedding(input.text)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Step 2: Perform the prediction
        with torch.no_grad():
            output = model(embedding_tensor)
            _, predicted = torch.max(output, dim=1)

        # Step 3: Map prediction to sentiment label
        sentiment_label = "POSITIVE" if predicted.item() == 1 else "NEGATIVE"

        return {"sentiment": sentiment_label}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# uvicorn app:app --reload
