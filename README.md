# RankingSystem

## Overview
RankingSystem is a Python-based application that performs sentiment analysis using a pre-trained transformer model to generate embeddings and an LSTM (Long Short-Term Memory) network for classification. The system is deployed as a FastAPI application with easy-to-use endpoints for analyzing the sentiment of text data.
## Features
- Sentiment analysis for text data
- Easy-to-use FastAPI endpoints
- Dockerized for easy deployment

## Dataset
The dataset used for this project can be found at the following link:
[Google Drive - Dataset](https://drive.google.com/drive/folders/1nUDEJy4wWoOlqggzsvTbOmQwmBv59Oml?usp=sharing)

This dataset contains the following files:
- **news.csv**: Raw news articles
- **processed_data_with_embeddings.pkl**: Processed dataset with embeddings
- **news_embeddings.npy**: Numpy file containing computed embeddings

## Installation

### Prerequisites
- Python 3.9+
- Docker (optional for containerized environment)

### Clone the Repository
```bash
git clone https://github.com//RankingSystem.git
cd RankingSystem
```

### Setting up the Environment
#### Create a Virtual Environment:
```bash
python -m venv .venv
```

#### Activate the Virtual Environment:
On Windows:
```bash
.\.venv\Scripts\activate
```
On macOS/Linux:
```bash
source .venv/bin/activate
```

#### Install Required Dependencies:
```bash
pip install -r requirements.txt
```

## Run the Application

### Run with Uvicorn:
```bash
uvicorn app:app --reload
```
The application will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Optional: Run with Docker
If you prefer a containerized environment, you can use Docker:
```bash
docker build -t ranking-system .
docker run -p 8000:8000 ranking-system
```

## API Endpoints
### **POST /predict/**
Analyzes the sentiment of the provided text.

#### **Request Body:**
```json
{
  "text": "<your_text_here>"
}
```

#### **Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.92
}
```

## Configuration
The application can be configured using the `config.yaml` file:
- **model**: Contains the model hyperparameters like `batch_size`, `learning_rate`, `num_epochs`, etc.
- **dataset**: Path to the dataset (e.g., `data/news.csv`).
- **remove_col**: Columns to remove from the dataset before training.

## Dockerfile
This project includes a Dockerfile to containerize the application. Here's a basic overview:
```dockerfile
FROM python:3.9
RUN apt-get update && apt-get install -y libgomp1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Contribution
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
