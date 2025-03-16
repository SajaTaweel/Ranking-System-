import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from logger import logger

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def train_lstm():
    try:
        logger.info("Loading embeddings and dataset...")
        embeddings = np.load("data/news_embeddings.npy")  # Load saved embeddings
        df = pd.read_pickle("data/processed_data_with_embeddings.pkl")

        assert len(embeddings) == len(df), "Mismatch between embeddings and dataset length."

        labels = torch.tensor(df["sentiment"].values, dtype=torch.long)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

        # Split data
        train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
            embeddings_tensor, labels,
            test_size=config["model"]["test_size"],
            random_state=config["model"]["random_state"]
        )

        # DataLoaders
        train_dataset = TensorDataset(train_embeddings, train_labels)
        test_dataset = TensorDataset(test_embeddings, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["model"]["batch_size"], shuffle=False)

        # Define LSTM model
        class LSTMClassifier(nn.Module):
            def __init__(self, input_dim):
                super(LSTMClassifier, self).__init__()
                self.lstm = nn.LSTM(
                    input_dim,
                    config["model"]["hidden_dim"],
                    num_layers=config["model"]["num_layers"],
                    batch_first=True
                )
                self.fc = nn.Linear(config["model"]["hidden_dim"], 2)  # Binary classification (Positive & Negative)

            def forward(self, x):
                x = x.unsqueeze(1)  # Add sequence length dimension
                lstm_out, _ = self.lstm(x)
                last_hidden_state = lstm_out[:, -1, :]
                return self.fc(last_hidden_state)

        input_dim = embeddings.shape[1]
        model = LSTMClassifier(input_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

        # Start MLflow Experiment
        mlflow.set_experiment(config["experiment_name"])

        with mlflow.start_run():
            mlflow.log_params({
                "batch_size": config["model"]["batch_size"],
                "learning_rate": config["model"]["learning_rate"],
                "num_epochs": config["model"]["num_epochs"],
                "hidden_dim": config["model"]["hidden_dim"],
                "num_layers": config["model"]["num_layers"]
            })

            # Training Loop
            logger.info("Starting LSTM training...")
            for epoch in range(config["model"]["num_epochs"]):
                model.train()
                total_loss = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                logger.info(f"Epoch [{epoch+1}/{config['model']['num_epochs']}], Loss: {avg_loss:.4f}")

            # Evaluation
            model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    predictions = torch.argmax(outputs, dim=1)
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())

            # Compute Metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"{config['negative_class']} - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-score: {f1[0]:.4f}")
            logger.info(f"{config['positive_class']} - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-score: {f1[1]:.4f}")

            # Log Metrics in MLflow
            mlflow.log_metrics({
                "test_accuracy": accuracy,
                f"precision_{config['negative_class']}": precision[0],
                f"recall_{config['negative_class']}": recall[0],
                f"f1_{config['negative_class']}": f1[0],
                f"precision_{config['positive_class']}": precision[1],
                f"recall_{config['positive_class']}": recall[1],
                f"f1_{config['positive_class']}": f1[1]
            })

            # Save Model
            torch.save(model.state_dict(), config["model_path"])
            mlflow.pytorch.log_model(model, "model")

            logger.info("LSTM model training completed and saved successfully.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
