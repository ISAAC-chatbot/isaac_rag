# utils/data_loader.py

import logging
from openai import OpenAI


def initialize_openai_client(api_key: str) -> OpenAI:
    try:
        client = OpenAI(api_key=api_key)
        logging.info("OpenAI 클라이언트가 성공적으로 초기화되었습니다.")
        return client
    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        raise