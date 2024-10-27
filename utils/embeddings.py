# utils/embeddings.py

import numpy as np
import logging

# 캐시 딕셔너리 초기화
embedding_cache = {}

def embed_text(client, texts):
    embeddings = []
    for text in texts:
        if text in embedding_cache:
            embeddings.append(embedding_cache[text])
            logging.info(f"임베딩 캐시에서 로드됨.")
        else:
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-ada-002"
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            embedding_cache[text] = embedding
            embeddings.append(embedding)
            logging.info(f"임베딩 생성 및 캐시됨.")
    return np.array(embeddings)
