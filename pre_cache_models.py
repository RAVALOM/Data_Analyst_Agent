# pre_cache_models.py
import os
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This is a good, general-purpose, and relatively small model.
MODEL_NAME = 'all-MiniLM-L6-v2'

def main():
    """
    Downloads and caches the specified sentence transformer model.
    The SENTENCE_TRANSFORMERS_HOME environment variable will direct
    where this model is stored.
    """
    logger.info(f"Attempting to download and cache model: {MODEL_NAME}")
    try:
        # This line will trigger the download and save it to the cache directory
        SentenceTransformer(MODEL_NAME)
        logger.info(f"Successfully cached model: {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to download model {MODEL_NAME}. Error: {e}")
        # Exit with a non-zero status code to fail the Docker build if caching fails
        exit(1)

if __name__ == "__main__":
    main()