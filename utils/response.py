# utils/response.py

import time
import logging
import langid
import json
import faiss
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from typing import List, Dict, Generator
from openai import OpenAI
from .embeddings import embed_text
from .search import search_faiss, search_bm25, search_bm25_faiss

def generate_response_with_context(client: OpenAI, query: str, context: List[dict], language: str):
    start_time = time.perf_counter()
    if not context:
        response = "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?" if language == 'ko' else "I'm sorry, I couldn't find any relevant information. Would you like to ask a different question?"
        logging.info("관련 정보를 찾을 수 없어 기본 응답을 반환합니다.")
        yield response
        return

    # 컨텍스트 텍스트 생성
    context_start = time.perf_counter()
    context_text = create_context_text(context)
    context_end = time.perf_counter()
    context_time = context_end - context_start
    logging.info(f"컨텍스트 텍스트 생성 시간: {context_time:.6f}초")

    # LLM에 전달할 메시지 생성
    messages = [
    {
        "role": "system",
        "content": f"""You are a friendly and knowledgeable university chatbot. Respond in a professional yet approachable manner, using polite '~요' endings, similar to the tone in the following examples:

        - "수강 변경 기간이 끝난 후에는 추가로 수강 신청이 어려워요. 데이터 처리 작업이 진행되기 때문이에요."
        - "수강 철회는 학사 포털에서 지정된 기간 동안만 가능해요. 보통 개강 후 5~6주차가 수강 철회 기간이에요."
        - "추가 정보가 필요하시면 학사지원팀에 문의해 보세요. 전화번호는 02-2123-2090, 2091, 2096, 2097입니다."

       If the query is not related to university information, respond with "대학교와 관련 없는 질문은 답변해드릴 수 없습니다."
       If no source URL is available, add '출처 : https://www.yonsei.ac.kr/sc/support/notice.jsp' at the end with the note '추가 정보가 없으니 해당 페이지를 참고해 주세요.' Otherwise, include the provided source URL only at the end of all responses, as '출처 : [url]'.
       Respond in {language}.
        """
    },
    {
        "role": "user",
        "content": f"Here is some background information with the source URLs provided:\n\n{context_text}\n\nBased on this information, could you help with the following question: {query}\n\nPlease make sure to include the relevant details and add the source URL only at the end of all responses."
    }
]



    # 메시지 구조 로깅
    logging.info(f"LLM에 전달되는 메시지: {json.dumps(messages, ensure_ascii=False, indent=2)[:1000]}...")  # 너무 길 경우 일부만 로깅

    try:
        # OpenAI API 호출
        api_start = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 올바른 모델 이름 사용
            messages=messages,
            max_tokens=500,  # 생성 토큰 수 제한
            stream=True  # 스트리밍 활성화
        )
        api_end = time.perf_counter()
        api_time = api_end - api_start
        logging.info(f"OpenAI API 호출 시간: {api_time:.6f}초")

        # 응답 포맷팅
        collected_response = ""
        # for chunk in response:
        #     chunk_message = chunk.choices[0].delta.content
        #     chunk_message = str(chunk_message)
        #     collected_response += chunk_message
        #     yield chunk_message  # 스트리밍 응답 반환
        for chunk in response:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is None:
                chunk_message = ''
            chunk_message = str(chunk_message)
            collected_response += chunk_message
            yield chunk_message  # 스트리밍 응답 반환


    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}")
        error_message = "죄송합니다. 응답을 생성하는 동안 오류가 발생했습니다. 다시 시도해 주세요." if language == 'ko' else "I'm sorry, an error occurred while generating the response. Please try again."
        yield error_message
        return

def create_context_text(context: List[dict], max_length: int = 10000) -> str:
    logging.info("컨텍스트 텍스트 생성 시작.")
    context_text = ""
    for doc in context:
        text = doc.get('merged_text', 'N/A')
        tables = doc.get('tables', 'N/A')
        doc_text = f"URL: {doc.get('url', 'Unknown URL')}\nText: {text}\nTables: {tables}\n"
        if len(context_text) + len(doc_text) > max_length:
            logging.info(f"문서 길이 : { len(context_text) + len(doc_text)}")
            remaining_length = max_length - len(context_text)
            if remaining_length > 0:
                truncated_doc_text = doc_text[:remaining_length]
                context_text += truncated_doc_text
                logging.info(f"문서 {doc.get('url', 'Unknown URL')} 일부를 잘라서 추가했습니다.")
            break
        context_text += doc_text
        # 개별 문서 텍스트 로깅 추가
        logging.info(f"추가된 문서: {doc_text[:100]}...")  # 전체 텍스트 대신 일부만 로깅
    logging.info("컨텍스트 텍스트 생성 완료.")
    logging.info(f"생성된 컨텍스트 텍스트 길이: {len(context_text)}")
    logging.info(f"생성된 컨텍스트 텍스트 내용: {context_text[:500]}...")  # 필요 시 일부만 로깅
    return context_text

def detect_language(text: str) -> str:
    lang, _ = langid.classify(text)
    logging.info(f"감지된 언어: {lang}")
    return lang

def chat_placeholder(client: OpenAI, index: faiss.Index, metadata: Dict, id_to_url: Dict, url_to_id: Dict, bm25: BM25Okapi, tokenizer: Okt, query: str, search_method: str, top_k_faiss: int = 5, bm25_top_k: int = 200, faiss_top_k: int = 5):
    try:
        overall_start = time.perf_counter()
        logging.info(f"새로운 쿼리 수신: {query}")

        # 언어 감지
        lang_start = time.perf_counter()
        lang = detect_language(query)
        lang_end = time.perf_counter()
        lang_time = lang_end - lang_start
        logging.info(f"언어 감지 완료: {lang} (시간: {lang_time:.6f}초)")

        context = []
        timings = {}
        if search_method == 'faiss':
            # FAISS만 사용
            faiss_start = time.perf_counter()
            faiss_result = search_faiss(client, index, metadata, id_to_url, query, top_k=top_k_faiss)
            faiss_end = time.perf_counter()
            faiss_time = faiss_end - faiss_start
            timings['faiss_time'] = faiss_time
            logging.info(f"FAISS 검색 시간: {faiss_time:.6f}초")
            context = [result["metadata"] for result in faiss_result["results"]]
            logging.info("FAISS 검색 결과를 사용하여 응답을 생성합니다.")
        elif search_method == 'bm25+faiss':
            # BM25 + FAISS 사용
            bm25_faiss_start = time.perf_counter()
            bm25_faiss_result = search_bm25_faiss(client, bm25, metadata, id_to_url, tokenizer, query, bm25_top_k=bm25_top_k, faiss_top_k=faiss_top_k)
            bm25_faiss_end = time.perf_counter()
            bm25_faiss_time = bm25_faiss_end - bm25_faiss_start
            timings['bm25_faiss_time'] = bm25_faiss_time
            logging.info(f"BM25 + FAISS 검색 시간: {bm25_faiss_time:.6f}초")
            context = [result["metadata"] for result in bm25_faiss_result["results"]]
            logging.info("BM25 + FAISS 검색 결과를 사용하여 응답을 생성합니다.")
        elif search_method == 'bm25':
            # BM25만 사용
            bm25_start = time.perf_counter()
            bm25_result = search_bm25(bm25, metadata, id_to_url, tokenizer, query, top_k=5)
            bm25_end = time.perf_counter()
            bm25_time = bm25_end - bm25_start
            timings['bm25_time'] = bm25_time
            logging.info(f"BM25 검색 시간: {bm25_time:.6f}초")
            context = [result["metadata"] for result in bm25_result["results"]]
            logging.info("BM25 검색 결과를 사용하여 응답을 생성합니다.")
        else:
            logging.error(f"알 수 없는 검색 방법: {search_method}")
            error_message = "죄송합니다. 내부 오류가 발생했습니다." if lang == 'ko' else "I'm sorry, an internal error occurred."
            yield error_message
            return

        # OpenAI를 사용하여 응답 생성
        response_start = time.perf_counter()
        response_generator = generate_response_with_context(client, query, context, lang)
        response_end = time.perf_counter()
        response_time = response_end - response_start
        timings['response_time'] = response_time
        logging.info(f"응답 생성 및 포맷팅 시간: {response_time:.6f}초")

        overall_end = time.perf_counter()
        total_time = overall_end - overall_start
        timings['total_time'] = total_time
        logging.info(f"전체 쿼리 처리 시간: {total_time:.6f}초")

        for chunk in response_generator:
            yield chunk

    except Exception as e:
        logging.error(f"오류가 발생했습니다: {str(e)}")
        error_message = "죄송합니다. 오류가 발생했습니다. 다시 시도해주세요." if 'lang' in locals() and lang == 'ko' else "I'm sorry, an error occurred. Please try again."
        yield error_message
        return
