# utils/logging_utils.py

import logging
import os
from datetime import datetime

def initialize_logging(log_file: str = 'logs/ai.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        encoding='utf-8'
    )
    logging.info("로깅 시스템 초기화 완료.")


def initialize_query_logging():
    # Create queries directory if it doesn't exist
    queries_dir = 'queries'
    os.makedirs(queries_dir, exist_ok=True)
    logging.info("쿼리 로깅 시스템 초기화 완료.")


def log_query(session_id: str, query: str, search_method: str):
    """
    Log query with session ID to a dedicated log file
    """
    queries_dir = 'queries'
    log_file = os.path.join(queries_dir, f'{session_id}_queries.log')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp}|{session_id}|{search_method}|{query}\n"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)


from datetime import datetime

def get_logger_for_user(user_id: str) -> logging.Logger:
    """
    유저별 고유 로거를 생성하여 반환
    """
    logger_name = f"logger_user_{user_id}"
    logger = logging.getLogger(logger_name)
    
    # 동일한 로거에 중복 핸들러 추가 방지
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # 로그 디렉토리 생성
        os.makedirs("logs/users", exist_ok=True)
        log_file = os.path.join("logs/users", f"{user_id}.log")
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.propagate = False  # 글로벌 로거로 메시지가 전파되지 않도록 설정

    return logger
