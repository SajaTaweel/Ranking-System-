import pandas as pd
from logger import logger


def extract(filename):

    try:
        logger.info(f"Extracting: {filename}")
        df = pd.read_csv(filename)
        logger.info(f"Extracted {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Failed to extract data: {e}")
        raise
