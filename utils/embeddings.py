# utils/embeddings.py

import logging
from openai import OpenAI
import time
import json



def generate_query_vector(client, input_text, original_query , logger=None):
    logger = logger or logging.getLogger(__name__)
    MODEL_NAME = "text-embedding-3-small"

        # Check if original_query exists in the JSON file
    json_file_path = "/home/ubuntu/multiturn/utils/query_embeddings.json"
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            existing_embeddings = json.load(f)
            if original_query in existing_embeddings:
                logger.info(f"Query '{original_query}' already exists in the embeddings file.")
                return existing_embeddings[original_query]
    except FileNotFoundError:
        logger.warning(f"JSON file '{json_file_path}' not found. Proceeding with embedding generation.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON file '{json_file_path}'. Proceeding with embedding generation.")


    MAX_RETRIES = 3
    try:
        def get_embedding_with_retry(client, text, retries=MAX_RETRIES):
            for attempt in range(1, retries + 1):
                try:
                    response = client.embeddings.create(model=MODEL_NAME, input=[text])
                    return response.data[0].embedding
                except Exception as e:
                    logger.error(f"[임베딩 오류] 시도 {attempt}/{retries}: {str(e)}")
                    if attempt == retries:
                        return None
                    logger.info("0.5초 후 재시도...")
                    time.sleep(0.5)

        embedding = get_embedding_with_retry(client, input_text)
        if embedding:
            logger.info(f"===임베딩 생성===")
            return embedding
        else:
            logger.info(f"===임베딩 생성 오류===")
            return None
    except Exception as e:
        logger.info(f"===임베딩 생성 오류 {e}===")
        return None