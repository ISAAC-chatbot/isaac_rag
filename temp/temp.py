import logging
import json
from openai import OpenAI
from utils.embeddings import generate_query_vector
import os
from utils.data_loader import initialize_openai_client

OPENAI_API_KEY="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"

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화 (API 키 필요)
openai_client =  initialize_openai_client(api_key=OPENAI_API_KEY)

# 쿼리 목록
queries = {
    "2025-1 학부 등록금 납부 일정": "2025-1 학부 등록금 납부 일정",
    "2025-1학기 추가등록 언제야?": "2025-1학기 추가등록 언제야?",
    "2025-1학기 수강신청 일정": "2025-1학기 수강신청 일정",
    "2025-1학기 S/U 제도에 대해 알려줘": "2025-1학기 S/U 제도에 대해 알려줘",
    "2025-1학기 수강과목 철회 중간고사 이전이야?": "2025-1학기 수강과목 철회 중간고사 이전이야?",
    "마일리지 총량에 대해서 과별로 정리해줘": "마일리지 총량에 대해서 과별로 정리해줘",
    "2025-1 송도학사 입사 일정": "2025-1 송도학사 입사 일정"
}

# 임베딩 매핑 딕셔너리
query_embeddings = {}

# 각 쿼리에 대해 임베딩 생성 및 매핑
for key, query in queries.items():
    logger.info(f"'{query}' 임베딩 생성 중...")
    embedding = generate_query_vector(client=openai_client, input_text=query, original_query=query, logger=logger)
    if embedding:
        query_embeddings[key] = embedding
        logger.info(f"'{key}' 임베딩 생성 완료.")
    else:
        logger.error(f"'{key}' 임베딩 생성 실패.")

# 결과 저장 (JSON 파일로 저장)
output_file = "query_embeddings.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(query_embeddings, f, ensure_ascii=False, indent=4)
logger.info(f"임베딩 결과가 '{output_file}'에 저장되었습니다.")

# 로드 시 사용 예
# with open(output_file, "r", encoding="utf-8") as f:
#     loaded_embeddings = json.load(f)
#     print(loaded_embeddings)
