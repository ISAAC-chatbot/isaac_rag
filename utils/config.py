import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOG_FILE = 'logs/search_comparison.log'
FAISS_INDEX_PATH = 'data/faiss_index.bin'
METADATA_PATH = 'data/metadata.json'
BM25_INDEX_PATH = 'data/bm25_index.pkl'
BM25_CORPUS_PATH = 'data/bm25_corpus.pkl'

# API 키가 없을 경우 예외 처리
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")