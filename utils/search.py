# utils/search.py

import requests
import json
import logging
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from .mappings import DEPARTMENT_MAPPINGS

# ====== 로깅 설정 추가 ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 터미널에 출력
        logging.FileHandler('/home/ubuntu/sang/logs/search.log')  # 파일에도 저장
    ]
)

# ====== 환경 설정 ======
# .env 파일 로드
load_dotenv('/home/ubuntu/multiturn/.env') 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_USER =  os.getenv('OPENSEARCH_USER')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD')
OPENSEARCH_GENERAL_INDEX = os.getenv('OPENSEARCH_GENERAL_INDEX')
OPENSEARCH_NOTICE_INDEX = os.getenv('OPENSEARCH_NOTICE_INDEX')


# ====== 하이브리드 검색 (BM25 + KNN) ======
# hybrid-pipeline-vector-focused
# hybrid-pipeline-text-focused 
# hybrid-pipeline-balanced 
# hybrid-pipeline-merged-priority

def hybrid_search(user_query_text, user_query_vector, top_k=5, pipeline_name="hybrid-pipeline-balanced", logger=None):

    logger = logger or logging.getLogger(__name__)

    """
    하이브리드 파이프라인(hybrid-pipeline-balanced)을 사용하여
    summarized_text(BM25) + vector(KNN) 의 스코어를 결합한 검색을 수행합니다.
    """
    print(f"검색 질의: {user_query_text, pipeline_name}")
    try:
        search_query = {
        "_source": ["url", "merged_text", "tables"],  # 가져올 필드
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "merged_text": user_query_text
                        }
                    },
                    {
                        "match": {
                            "merged_text": user_query_text
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
        url = f"{OPENSEARCH_ENDPOINT}/{OPENSEARCH_GENERAL_INDEX}/_search?search_pipeline={pipeline_name}"
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
    
def notice_search(query_text, topic, top_k=5, logger=None):
    """토픽별 공지사항 검색"""
    logger = logger or logging.getLogger(__name__)
    
    # topic이 이미 DEPARTMENT_MAPPINGS의 키값인 경우 그대로 사용
    source = topic if topic in DEPARTMENT_MAPPINGS else None
    logger.info(f"Initial topic: {topic}")
    
    # topic이 value에 있는 경우 해당하는 키값을 source로 사용
    if not source:
        for dept, keywords in DEPARTMENT_MAPPINGS.items():
            if topic in keywords:
                source = dept
                logger.info(f"Found matching department: {dept} for topic: {topic}")
                break
    
    if not source:
        logger.warning(f"Topic {topic}에 대한 매핑된 source를 찾을 수 없습니다.")
        return []
    
    try:
        search_query = {
            "_source": ["url", "title", "content", "source", "createdDate"],
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["title", "content"],
                                "type": "most_fields",
                                "operator": "OR",
                                "analyzer": "nori"
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {
                                "source": source
                            }
                        }
                    ]
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"createdDate": {"order": "desc"}}
            ]
        }
        
        logger.info(f"Search query: {json.dumps(search_query, indent=2, ensure_ascii=False)}")

        url = f"{OPENSEARCH_ENDPOINT}/{OPENSEARCH_NOTICE_INDEX}/_search"
        response = requests.post(
            url,
            auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_query)
        )

        if response.status_code != 200:
            logger.error(f"공지사항 검색 실패: {response.status_code}")
            logger.error(f"응답: {response.text}")
            return []

        result = response.json()
        documents = []
        
        """
        for hit in result.get("hits", {}).get("hits", []):
            doc = hit.get("_source", {})
            # title과 content만 통합
            merged_text = (
                f"{doc.get('title', '')}\n"
                f"{doc.get('content', '')}\n"
                f"{doc.get('createdDate', '')}\n"
                f"{doc.get('source', '')}"
            )
            
            documents.append({
                "url": doc.get("url", ""),
                "merged_text": merged_text
            })
        """
        
        for hit in result.get("hits", {}).get("hits", []):
            doc = hit.get("_source", {})
            # _create_context 형식에 맞게 변환하되, URL을 "출처:" 형식으로 포함
            formatted_doc = {
                "url": doc.get("url", ""),
                "merged_text": (
                    f"출처: {doc.get('url', '')}\n"  # 명시적으로 "출처:" 포함
                    f"{doc.get('title', '')}\n"
                    f"{doc.get('content', '')}\n"
                    f"{doc.get('createdDate', '')}\n"
                    f"{doc.get('source', '')}"
                ),
                "tables": "N/A"
            }
            documents.append(formatted_doc)
            
        logger.info(f"공지사항 검색 결과 수: {len(documents)}")
        return documents

    except Exception as e:
        logger.error(f"공지사항 검색 중 오류 발생: {str(e)}")
        return []

def _search_documents_notice(self, state: dict) -> dict:
    """공지사항 문서 검색"""
    try:
        query_text = state.get("query", "")
        topics = state.get("topics", [])
        
        if not topics:
            return {"notice_documents": []}
            
        # 모든 토픽에 대해 검색 수행
        all_notice_documents = []
        for topic in topics:
            notice_documents = notice_search(
                query_text=query_text,
                topic=topic,
                top_k=5,
                logger=self.logger
            )
            all_notice_documents.extend(notice_documents)
            
        return {"notice_documents": all_notice_documents}
        
    except Exception as e:
        self.logger.error(f"공지사항 검색 중 오류 발생: {str(e)}")
        return {"notice_documents": []}


