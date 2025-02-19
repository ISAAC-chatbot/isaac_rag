import requests
import logging
import os
import json
import logging
import time
import gc
from openai import OpenAI
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
import re

OPENAI_API_KEY = "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"  # OpenAI API Key

OPENSEARCH_ENDPOINT = "https://search-issac-pd5jfsetumxcg6ciyfpowgi5fi.ap-northeast-2.es.amazonaws.com"
OPENSEARCH_USER = "issac"
OPENSEARCH_PASSWORD = "StrongPass!2025"
OPENSEARCH_INDEX = "ai_data"

MODEL_NAME = "text-embedding-3-small"

# 로그 설정
LOG_FILE = "opensearch_indexing.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()



def index_to_opensearch(doc):
    """
    OpenSearch에 문서(공지)를 실시간으로 업로드하는 함수
    doc: 파싱 결과(dict)
    """
    api_url = f"{OPENSEARCH_ENDPOINT}/{OPENSEARCH_INDEX}/_doc"
    headers = {"Content-Type": "application/json"}

    result_log = {"status": "unknown", "url": doc.get("url"), "response": None}

    try:
        response = requests.post(api_url, auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD), headers=headers, json=doc)
        if response.status_code in (200, 201):
            _id = response.json()["_id"]
            result_log["status"] = "success"
            result_log["response"] = {"_id": _id}
            logger.info(f"Successfully inserted document! ID: {_id}")
        else:
            result_log["status"] = "failure"
            result_log["response"] = {"status_code": response.status_code, "response_text": response.text}
            logger.error(f"Failed to insert document. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        result_log["status"] = "error"
        result_log["response"] = {"exception": str(e)}
        logger.error(f"Exception occurred while inserting document: {e}")

    # 결과 로그를 파일에 추가 저장
    with open("result_log.json", "a") as log_file:
        log_file.write(f"{result_log}\n")


def add_new_field(item_data: dict) -> dict:
    """
    item_data(dict)에 원하는 처리를 하여(예: GPT 요약) 새로운 필드를 추가해 반환.
    여기서는 'summarized_text' 추가 예시.
    """
    merged_text = item_data.get("merged_text", "")
    if not merged_text.strip():
        # 만약 merged_text가 비었다면 굳이 API 호출하지 않고 빈 값만 넣어둠
        item_data["summarized_text"] = ""
        return item_data
    
    try:
        system_prompt = """
        You are a specialized AI assistant proficient in extracting key metadata
        and summarizing text. You must ensure the final output includes these fields:
        title, summary, keywords, expected_queries, purpose, document_excerpt, and date.

        Important note for 'date':
        - If the text includes semester information (e.g., '2학기', 'Fall semester') and an explicit date, combine them into a single string in Korean (e.g., "2024년 2학기 8월 17일").
        - If the text includes only a date but no semester, output just the date in Korean format (e.g., "2024년 8월 17일").
        - If neither a specific date nor semester is available, leave it empty ("").
        """

        user_prompt = """
        Act as a prompt engineer. From the input text, extract the following fields:
        1) "title": A short, representative title for emphasizing search weight or quick reference (Korean).
        2) "summary": A concise 2-3 sentence summary capturing the key context (Korean).
        3) "keywords": Relevant concepts to aid in search indexing (Korean).
        4) "expected_queries": Five potential user queries, anticipating vector/BM25 searches (Korean).
        5) "purpose": The document’s intent or target audience (Korean).
        6) "document_excerpt": Direct quotes or crucial data (Korean).
        7) "date": If a semester and date are found, combine them (e.g., "2024년 2학기 8월 17일"). If only a date, "2024년 8월 17일". Otherwise, leave it empty.

        Important:
        1) The tags (title, summary, keywords, expected_queries, purpose, document_excerpt, date) must remain in English.
        2) The content for each tag must be written in Korean.
        3) Do not include any additional commentary or fields; strictly produce the requested JSON structure.
        4) If any field is not applicable, leave it blank or provide a minimal placeholder in Korean.
        5) Ensure exactly five items are listed under "expected_queries".
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": merged_text},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",  # 실제 사용 모델명으로 교체
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        gpt_output = response.choices[0].message.content.strip()
        item_data["summarized_text"] = gpt_output
    except Exception as e:
        logger.error(f"OpenAI API 요청 실패: {e}")
        item_data["summarized_text"] = ""

    return item_data



doc = {

    "url": "https://yicdorm.yonsei.ac.kr/board.asp?mid=m05_01&act=view&bid=1&idx=43033&page=1",
    "merged_text": """
제목 : 2025학년도 1학기 송도학사 입사 시설물 점검표, 객실 정보, 주의사항, 기숙사 방에서 전열기 사용 금지 안내
6. 시설물 점검표 작성 (필수)
  가. 기간: 2025. 2. 28.(금) ~ 3. 7.(금)
  나. 방법: 입사 시 시설물 점검표를 수령하여 배정된 객실 확인 후 점검표 작성, 이후 1학사
는 A동, 2학사는 D동 경비실에 제출
      ※ 시설물 파손 시 변상 기준이 되므로 입사 후 반드시 작성하시기 바랍니다.
 
 7. 객실 정보
  가. 내부 시설: 책상, 의자, 옷장(280×2100), 신발장, 침대 매트리스 (1000×2000×200), 
개별 냉난방 시설, 샤워실, 화장실, LAN포트 및 공유기
  나. 개인 준비물: 신분증, 매트리스 커버 (싱글 사이즈, 1000×2000×200), 이불, 베개, 세
면도구, 옷걸이, 슬리퍼, 그 외 기타 개인용품
※ 공동 생활을 하는 곳이므로 개인 소지품은 적정 수준으로 준비하시기 바랍니다.

다. 기숙사 방에서 전열기 사용 금지 
     
사용 가능 : 헤어드라이어, 고데기
사용 불가능 : 전기 장판/방석/담요, 난로, 냉장고, 다리미 등 

사용 가능 기기 외의 모든 전열기

  라. 애완동물 반입 금지
  마. 택배 수령 방법: 각 학사 택배실 또는 무인택배함(smilebox)에서 본인이 직접 수령
      ※ 분실의 위험이 있으니 귀중품은 직접 가져오시기 바랍니다.
      ※ 배정된 기숙사 동/호수 확인 후 택배 발송 바랍니다.
      ※ 주소 : 인천광역시 연수구 송도과학로 85 연세대학교 국제캠퍼스 송도0학사 0동 000호

      8. 주의사항
  가. 신청 완료 이후에는 성향, 방타입 등 세부 사항의 변경이 불가능합니다.
  나. 지정된 퇴사 기간까지 정당한 사유 없이 퇴사를 지연하거나 물건을 방치하는 등 퇴사 
절차를 미이행한 때에는 경고 조치와 함께 추가 비용이 부과될 수 있습니다.
  다. 학기중 휴학, 자퇴시 기숙사를 이용할 수 없습니다(휴학, 자퇴 신청과 동시에 퇴사).
  라. 기숙사 퇴사시 반드시 출입카드를 반납해야합니다. 카드 미반납, 시설물 파손 등의 경우 
해당 실비를 납부해야하며, 고지된 기간 내에 해당 금액을 납부하지 않을 경우 연세대
학교 제증명서 발급이 중단될 수 있습니다.
      }
    }
  }
}

3) 기숙사비는 9~10분위 실납입액을 우선 납부하며, 소득분위에 따라 기숙사장학금을 차등 
적용하여 7월 중순 해당 차액을 본인의 국내계좌로 지급합니다. 소득분위는 한국장학재
단의 국가장학금 소득부누이 자료를 활용하오니, 반드시 국가장학금을 신청해 주십시오
(한국장학재단 국가장학금 관련 문의 → http://www.kosaf.go.kr)
      예)·3인실 9~10분위의 경우 장학금 지급 없음(92,000원 사전 감면)
         ·3인실 2~8분위의 경우 장학금 32,000원(124,000원-92,000원) 7월 중순 지급 예정
         ·3인실 0~1분위의 경우 장학금 444,000원(536,000원-92,000원) 7월 중순 지급 예정
        ※ 기숙사 장학금의 환불은 연세포털서비스에 등록된 본인 명의의 국내 계좌로 반환
되오니, 반드시 연세포털서비스에 계좌 등록을 해주시기 바랍니다.
        ※ 연세한마음전형 입학자는 신청 시 0~1분위에 해당하는 장학금이 사전 감면되며, 그 
외 전형으로 입학한 국민기초생활보장수급자는 수급자증명서(좌측 상단에 학번/이름 기
재)를 2025. 2. 21.(금)까지 residence@yonsei.ac.kr 로 제출해야 사전 감면됩니다.
  바. 기숙사비에 식사 비용은 포함되어 있지 않습니다.
    """,
    "images": [],
    "files": [],
    "tables": [],
    "isnotice": False,
    "subject": "연세대전반",
    "vector" :""
}


def generate_query_vector(client, input_text,logger=None):
    logger = logger or logging.getLogger(__name__)
    MODEL_NAME = "text-embedding-3-small"
    MAX_RETRIES = 3
    try:
        def get_embedding_with_retry(client, text, retries=MAX_RETRIES):
            for attempt in range(1, retries + 1):
                try:
                    response = client.embeddings.create(model=MODEL_NAME, input=[text])
                    return response.data[0].embedding
                except Exception as e:
                    logger.error(f"[임베딩 오류] 시도 {attempt}/{retries}: {str(e)}")
                    if attempt == retries:
                        return None
                    logger.info("0.5초 후 재시도...")
                    time.sleep(0.5)

        embedding = get_embedding_with_retry(client, input_text)
        if embedding:
            logger.info(f"===임베딩 생성===")
            return embedding
        else:
            logger.info(f"===임베딩 생성 오류===")
            return None
    except Exception as e:
        logger.info(f"===임베딩 생성 오류 {e}===")
        return None
    

client = OpenAI(api_key=OPENAI_API_KEY)

add_new_field(doc)

text=doc['summarized_text']

doc['vector'] = generate_query_vector(client, text ,logger)

index_to_opensearch(doc)

print(doc)