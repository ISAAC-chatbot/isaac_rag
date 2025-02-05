# utils/data_loader.py

import logging
from openai import OpenAI

from utils.logging_utils import (
    initialize_logging, 
    initialize_query_logging)
import os
from dotenv import load_dotenv


# 현재 스크립트가 실행되는 디렉토리의 부모 디렉토리 찾기
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# .env 파일의 전체 경로 지정
dotenv_path = os.path.join(base_dir, ".env")

# .env 파일을 로드합니다
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)


def initialize_openai_client(api_key: str) -> OpenAI:
    try:
        client = OpenAI(api_key=api_key)
        logging.info("OpenAI 클라이언트가 성공적으로 초기화되었습니다.")
        return client
    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        raise

def initialize_client():
    # 초기화 함수 호출
    initialize_logging(log_file='logs/ai_global.log')
    # initialize_search_time_file()
    initialize_query_logging()

    # OpenAI 클라이언트 초기화
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수 'OPENAI_API_KEY'를 설정해주세요.")

        if not OPENAI_API_KEY.startswith('sk-'):
            raise ValueError("잘못된 API 키 형식입니다. OpenAI API 키는 'sk-'로 시작해야 합니다.")

        return initialize_openai_client(api_key=OPENAI_API_KEY)

    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        raise