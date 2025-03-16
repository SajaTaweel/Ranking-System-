import os
import yaml
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from logger import logger
from embedding import get_embedding

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special characters and punctuation
    words = word_tokenize(text)  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_data(df):
    try:
        logger.info("Data Preprocessing...")

        # Remove unwanted columns
        data = df.drop(columns=config['remove_col'], axis=1)

        # Drop duplicate news entries and missing values
        data.drop_duplicates(subset=["news"], inplace=True)
        data.dropna(subset=["news"], inplace=True)

        # Apply text cleaning
        data["news"] = data["news"].apply(clean_text)

        # Encode sentiment labels
        sentiment_mapping = {"POSITIVE": 1, "NEGATIVE": 0}
        data["sentiment"] = data["sentiment"].map(sentiment_mapping)

        # Check if embeddings are already saved
        embeddings_file = "data/news_embeddings.npy"
        if os.path.exists(embeddings_file):
            logger.info("Loading saved embeddings...")
            embeddings_list = np.load(embeddings_file, allow_pickle=True)
            # Assign the loaded embeddings back to the DataFrame
            embeddings = [embeddings_list[i] for i in range(embeddings_list.shape[0])]
            data["news"] = pd.Series(embeddings, index=data.index)
        else:
            logger.info("Generating embeddings for news articles...")
            data["news"] = data["news"].apply(get_embedding)
            embeddings_list = data["news"].tolist()
            # Save the embeddings list to a .npy file for future use
            np.save("data/news_embeddings.npy", embeddings_list)

            df.to_pickle("data/processed_data_with_embeddings.pkl")

        logger.info("Preprocessing completed successfully!")
        return data

    except Exception as e:
        logger.error(f"Failed to preprocess the data: {e}")
        raise
