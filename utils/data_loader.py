# utils/data_loader.py
import os
import json
import faiss
import pickle
import logging
from konlpy.tag import Okt
from openai import OpenAI
from .config import FAISS_INDEX_PATH, METADATA_PATH, BM25_INDEX_PATH

def initialize_openai_client(api_key: str) -> OpenAI:
    try:
        client = OpenAI(api_key=api_key)
        logging.info("OpenAI 클라이언트가 성공적으로 초기화되었습니다.")
        return client
    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        raise

def load_faiss_index():
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            logging.info("FAISS 인덱스를 파일에서 불러왔습니다.")

            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata = {int(k): v for k, v in metadata.items()}
            logging.info("메타데이터를 파일에서 불러왔습니다.")
            return index, metadata
        else:
            raise FileNotFoundError("faiss_index.bin 또는 metadata.json 파일이 존재하지 않습니다.")
    except Exception as e:
        logging.error(f"FAISS 인덱스 또는 메타데이터 로드 실패: {str(e)}")
        raise

def load_or_create_bm25(metadata: dict, tokenizer: Okt):
    if os.path.exists(BM25_INDEX_PATH):
        try:
            with open(BM25_INDEX_PATH, 'rb') as f:
                bm25 = pickle.load(f)
            logging.info("BM25 인덱스를 파일에서 불러왔습니다.")
            return bm25
        except Exception as e:
            logging.error(f"BM25 인덱스 로드 실패: {str(e)}")
            raise
    else:
        # BM25 코퍼스 생성 및 인덱스 초기화 코드
        from rank_bm25 import BM25Okapi
        import time

        start_time = time.perf_counter()
        try:
            logging.info("BM25 코퍼스 생성을 시작합니다.")
            bm25_corpus = []
            for idx in metadata:
                text = metadata[idx].get("merged_text", "")
                tables = metadata[idx].get("tables", [])
                tables_text = " ".join([str(table) for table in tables])
                combined_text = text + " " + tables_text
                tokens = tokenizer.morphs(combined_text)
                bm25_corpus.append(tokens)
            logging.info("BM25 코퍼스 생성을 완료했습니다.")
        except Exception as e:
            logging.error(f"BM25 코퍼스 생성 중 오류 발생: {str(e)}")
            raise
        end_time = time.perf_counter()
        bm25_corpus_time = end_time - start_time
        logging.info(f"BM25 코퍼스 생성 시간: {bm25_corpus_time:.6f}초")

        # BM25 인덱스 초기화
        start_time = time.perf_counter()
        try:
            bm25 = BM25Okapi(bm25_corpus)
            logging.info("BM25 인덱스를 초기화했습니다.")
        except Exception as e:
            logging.error(f"BM25 인덱스 초기화 실패: {str(e)}")
            raise
        end_time = time.perf_counter()
        bm25_init_time = end_time - start_time
        logging.info(f"BM25 인덱스 초기화 시간: {bm25_init_time:.6f}초")

        # BM25 인덱스 저장
        try:
            with open(BM25_INDEX_PATH, 'wb') as f:
                pickle.dump(bm25, f)
            logging.info("BM25 인덱스를 파일에 저장했습니다.")
        except Exception as e:
            logging.error(f"BM25 인덱스 저장 실패: {str(e)}")
            raise

        return bm25