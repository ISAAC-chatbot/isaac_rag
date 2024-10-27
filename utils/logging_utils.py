# utils/logging_utils.py

import logging
import os


def initialize_search_time_file(search_time_file: str = "search_time_comparison.txt"):
    if not os.path.exists(search_time_file):
        with open(search_time_file, "w", encoding='utf-8') as f:
            f.write("Search Time Comparison\n")
            f.write("="*50 + "\n")
        logging.info("검색 시간 비교 파일 초기화 완료.")


def initialize_logging(log_file: str = 'logs/search_comparison.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        encoding='utf-8'
    )
    logging.info("로깅 시스템 초기화 완료.")