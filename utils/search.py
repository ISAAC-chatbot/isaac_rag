# utils/search.py

import time
import numpy as np
import faiss
import logging
from .embeddings import embed_text
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

def search_faiss(client, index, metadata, id_to_url, query, top_k=5):
    logging.info("FAISS 검색 함수 호출")
    timings = {}
    try:
        # 쿼리 임베딩 생성
        embed_start = time.perf_counter()
        query_embedding = embed_text(client, [query])[0]
        embed_end = time.perf_counter()
        timings['embedding_creation'] = embed_end - embed_start
        logging.info(f"쿼리 임베딩 생성 시간: {timings['embedding_creation']:.6f}초")

        # FAISS 검색 수행
        search_start = time.perf_counter()
        distances, indices = index.search(np.expand_dims(query_embedding, axis=0), top_k)
        search_end = time.perf_counter()
        timings['faiss_search'] = search_end - search_start
        logging.info(f"FAISS 검색 시간: {timings['faiss_search']:.6f}초")

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in metadata:
                results.append({
                    "distance": dist,
                    "metadata": metadata[idx],
                    "url": id_to_url.get(idx, "Unknown URL")
                })

        return {
            "results": results,
            "closest_distance": distances[0][0] if distances.size > 0 else None,
            "timings": timings
        }
    except Exception as e:
        logging.error(f"FAISS 검색 실패: {str(e)}")
        raise

def search_bm25(bm25, metadata, id_to_url, tokenizer, query, top_k=5):
    logging.info("BM25 검색 함수 호출")
    timings = {}
    try:
        # 질문 토큰화
        token_start = time.perf_counter()
        tokens = tokenizer.morphs(query)
        token_end = time.perf_counter()
        timings['tokenization'] = token_end - token_start
        logging.info(f"쿼리 토큰화 시간: {timings['tokenization']:.6f}초")

        # BM25 점수 계산
        score_start = time.perf_counter()
        scores = bm25.get_scores(tokens)
        score_end = time.perf_counter()
        timings['bm25_score'] = score_end - score_start
        logging.info(f"BM25 점수 계산 시간: {timings['bm25_score']:.6f}초")

        # 상위 인덱스 정렬
        sort_start = time.perf_counter()
        top_indices = np.argsort(scores)[::-1][:top_k]
        sort_end = time.perf_counter()
        timings['sort'] = sort_end - sort_start
        logging.info(f"BM25 상위 인덱스 정렬 시간: {timings['sort']:.6f}초")

        results = []
        for idx in top_indices:
            if idx in metadata:
                results.append({
                    "score": scores[idx],
                    "metadata": metadata[idx],
                    "url": id_to_url.get(idx, "Unknown URL")
                })

        return {
            "results": results,
            "score": scores[top_indices[0]] if len(top_indices) > 0 else None,
            "timings": timings
        }
    except Exception as e:
        logging.error(f"BM25 검색 실패: {str(e)}")
        raise

def search_bm25_faiss(client, bm25, metadata, id_to_url, tokenizer, query, bm25_top_k=10, faiss_top_k=5, dimension=1536):
    timings = {}
    logging.info("BM25 + FAISS 검색 함수 호출")
    try:
        # BM25 검색
        bm25_start = time.perf_counter()
        bm25_results = search_bm25(bm25, metadata, id_to_url, tokenizer, query, top_k=bm25_top_k)
        bm25_end = time.perf_counter()
        timings['bm25_total'] = bm25_end - bm25_start
        logging.info(f"BM25 총 검색 시간: {timings['bm25_total']:.6f}초")

        bm25_docs = bm25_results["results"]

        if not bm25_docs:
            logging.info("BM25 검색 결과가 없습니다.")
            return {
                "results": [],
                "timings": timings
            }

        # BM25 상위 문서들의 텍스트와 테이블을 결합
        combined_texts = []
        for doc in bm25_docs:
            text = doc['metadata'].get("merged_text", "")
            tables = doc['metadata'].get("tables", [])
            tables_text = " ".join([str(table) for table in tables])
            combined_text = text + " " + tables_text
            combined_texts.append(combined_text)

        # BM25 상위 문서들의 임베딩 생성
        embed_start = time.perf_counter()
        doc_embeddings = embed_text(client, combined_texts)
        embed_end = time.perf_counter()
        timings['embedding_creation'] = embed_end - embed_start
        logging.info(f"BM25 상위 문서 임베딩 생성 시간: {timings['embedding_creation']:.6f}초")

        # 쿼리 임베딩 생성
        query_embedding = embed_text(client, [query])[0]
        logging.info("쿼리 임베딩 생성 완료.")

        # FAISS 인덱스 생성 및 검색
        faiss_start = time.perf_counter()
        temp_index = faiss.IndexFlatIP(dimension)  # Inner Product 기반 인덱스
        faiss.normalize_L2(doc_embeddings)
        faiss.normalize_L2(np.expand_dims(query_embedding, axis=0))
        temp_index.add(doc_embeddings)
        distances, indices = temp_index.search(np.expand_dims(query_embedding, axis=0), faiss_top_k)
        faiss_end = time.perf_counter()
        timings['faiss_search'] = faiss_end - faiss_start
        logging.info(f"FAISS 검색 시간: {timings['faiss_search']:.6f}초")

        # FAISS 검색 결과 수집
        faiss_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(bm25_docs):
                doc = bm25_docs[idx]
                faiss_results.append({
                    "distance": dist,
                    "metadata": doc['metadata'],
                    "url": doc['url']
                })

        return {
            "results": faiss_results,
            "timings": timings
        }

    except Exception as e:
        logging.error(f"BM25 + FAISS 검색 실패: {str(e)}")
        raise