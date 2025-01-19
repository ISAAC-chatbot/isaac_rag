# utils/search.py

import requests
import json
import logging
from openai import OpenAI
import time
import os
from dotenv import load_dotenv


# ====== 환경 설정 ======
# .env 파일 로드
load_dotenv('/home/ubuntu/multiturn/.env') 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_USER =  os.getenv('OPENSEARCH_USER')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX')


# ====== 하이브리드 검색 (BM25 + KNN) ======
# hybrid-pipeline-vector-focused 1:9
# hybrid-pipeline-text-focused 9:1
# hybrid-pipeline-balanced 5:5

def hybrid_search(user_query_text, user_query_vector, top_k=5, pipeline_name="hybrid-pipeline-balanced", logger=None):

    logger = logger or logging.getLogger(__name__)

    """
    하이브리드 파이프라인(hybrid-pipeline-balanced)을 사용하여
    summarized_text(BM25) + vector(KNN) 의 스코어를 결합한 검색을 수행합니다.
    """
    try:
        search_query = {
            "_source": ["url", "merged_text", "tables"],  # 가져올 필드
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "summarized_text": user_query_text
                            }
                        },
                        {
                            "knn": {
                                "vector": {
                                    "vector": user_query_vector,
                                    "k": top_k
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

        # search_pipeline 파라미터로 하이브리드 파이프라인 지정
        url = f"{OPENSEARCH_ENDPOINT}/{OPENSEARCH_INDEX}/_search?search_pipeline={pipeline_name}"
        response = requests.post(
            url,
            auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_query)
        )

        logger.info(f"\n \n response : {response}")

        if response.status_code != 200:
            logger.error(f"검색 실패: {response.status_code} - {response.text}")
            return []

        result = response.json()
        # logger.info(f"검색 성공: {json.dumps(result, indent=2, ensure_ascii=False)}")

        """
        # 상위 점수 출력
        logger.info("검색된 상위 문서 점수:")
        for idx, hit in enumerate(result.get("hits", {}).get("hits", []), start=1):
            score = hit.get("_score", "N/A")
            doc_id = hit.get("_id", "Unknown ID")
            logger.info(f"[{idx}] 문서 ID: {doc_id}, 점수: {score}")
        """

        # documents 리스트 생성
        documents = []
        for hit in result.get("hits", {}).get("hits", []):
            doc = hit.get("_source", {})
            documents.append(doc)

        return documents

    except Exception as e:
        logger.error(f"검색 중 오류 발생: {str(e)}")
        return []