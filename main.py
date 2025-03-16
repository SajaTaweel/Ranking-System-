import yaml
from logger import logger
from extract import extract
from feature_engineering import preprocess_data
from training_lstm import train_lstm
import uvicorn


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
def main():
    try:
        logger.info("Starting pipeline...")

        df = extract(config["dataset"])
        data = preprocess_data(df)
        print(data["sentiment"].unique())
        print(data.head())

        # Train data
        train_lstm()
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Failed to finish the pipeline process: {e}")


if __name__ == '__main__':
   uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
