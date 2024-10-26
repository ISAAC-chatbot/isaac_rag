from bs4 import BeautifulSoup
import faiss
import numpy as np
from openai import OpenAI
import schedule
import time
from typing import List, Dict, Generator
import threading
import langid
import os
import json
import gradio as gr
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
import pickle
import logging
import hashlib

# 캐시 딕셔너리 초기화
embedding_cache = {}

# 공통 함수 정의

def initialize_logging(log_file: str = 'search_comparison.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO, # INFO DEBUG
        format='%(asctime)s:%(levelname)s:%(message)s',
        encoding='utf-8'
    )
    logging.info("로깅 시스템 초기화 완료.")

def initialize_search_time_file(search_time_file: str = "search_time_comparison.txt"):
    if not os.path.exists(search_time_file):
        with open(search_time_file, "w", encoding='utf-8') as f:
            f.write("Search Time Comparison\n")
            f.write("="*50 + "\n")
        logging.info("검색 시간 비교 파일 초기화 완료.")

def initialize_openai_client(api_key: str) -> OpenAI:
    try:
        client = OpenAI(api_key=api_key)
        logging.info("OpenAI 클라이언트가 성공적으로 초기화되었습니다.")
        return client
    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        raise

def load_faiss_index(faiss_index_path: str, metadata_path: str) -> (faiss.Index, Dict):
    try:
        if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
            index = faiss.read_index(faiss_index_path)
            logging.info("FAISS 인덱스를 파일에서 불러왔습니다.")

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata = {int(k): v for k, v in metadata.items()}
            logging.info("메타데이터를 파일에서 불러왔습니다.")
            return index, metadata
        else:
            raise FileNotFoundError("faiss_index.bin 또는 metadata.json 파일이 존재하지 않습니다.")
    except Exception as e:
        logging.error(f"FAISS 인덱스 또는 메타데이터 로드 실패: {str(e)}")
        raise

def load_or_create_bm25(bm25_corpus_path: str, bm25_index_path: str, metadata: Dict, tokenizer: Okt) -> BM25Okapi:
    if os.path.exists(bm25_index_path):
        try:
            with open(bm25_index_path, 'rb') as f:
                bm25 = pickle.load(f)
            logging.info("BM25 인덱스를 파일에서 불러왔습니다.")
            return bm25
        except Exception as e:
            logging.error(f"BM25 인덱스 로드 실패: {str(e)}")
            raise
    else:
        # BM25 코퍼스 생성
        start_time = time.perf_counter()
        try:
            logging.info("BM25 코퍼스 생성을 시작합니다.")
            bm25_corpus = []
            for idx in metadata:
                text = metadata[idx].get("merged_text", "")
                tables = metadata[idx].get("tables", [])
                tables_text = " ".join([str(table) for table in tables])
                combined_text = text + " " + tables_text
                tokens = tokenizer.morphs(combined_text)
                bm25_corpus.append(tokens)
            logging.info("BM25 코퍼스 생성을 완료했습니다.")
        except Exception as e:
            logging.error(f"BM25 코퍼스 생성 중 오류 발생: {str(e)}")
            raise
        end_time = time.perf_counter()
        bm25_corpus_time = end_time - start_time
        logging.info(f"BM25 코퍼스 생성 시간: {bm25_corpus_time:.6f}초")

        # BM25 인덱스 초기화
        start_time = time.perf_counter()
        try:
            bm25 = BM25Okapi(bm25_corpus)
            logging.info("BM25 인덱스를 초기화했습니다.")
        except Exception as e:
            logging.error(f"BM25 인덱스 초기화 실패: {str(e)}")
            raise
        end_time = time.perf_counter()
        bm25_init_time = end_time - start_time
        logging.info(f"BM25 인덱스 초기화 시간: {bm25_init_time:.6f}초")

        # BM25 인덱스 저장
        try:
            with open(bm25_index_path, 'wb') as f:
                pickle.dump(bm25, f)
            logging.info("BM25 인덱스를 파일에 저장했습니다.")
        except Exception as e:
            logging.error(f"BM25 인덱스 저장 실패: {str(e)}")
            raise

        return bm25

def detect_language(text: str) -> str:
    lang, _ = langid.classify(text)
    logging.info(f"감지된 언어: {lang}")
    return lang

def create_context_text(context: List[Dict], max_length: int = 15000) -> str:
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

def embed_text(client: OpenAI, texts: List[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        if text in embedding_cache:
            embeddings.append(embedding_cache[text])
            logging.info(f"임베딩 캐시에서 로드됨.")
        else:
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-ada-002"
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            embedding_cache[text] = embedding
            embeddings.append(embedding)
            logging.info(f"임베딩 생성 및 캐시됨.")
    return np.array(embeddings)

# 응답 생성 함수

def generate_response_with_context(client: OpenAI, query: str, context: List[Dict], language: str):
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
        {"role": "system", "content": f"You are a friendly and supportive assistant. Please use the context below to create a response that is both informative and reassuring. Respond in {language}."},
        {"role": "user", "content": f"Here is some background information with the source URLs provided:\n\n{context_text}\n\nBased on this information, could you help with the following question: {query}\n\nPlease make sure to mention the relevant details and include the source URL in your response for clarity."}
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



# FAISS 검색 함수
def search_faiss(client: OpenAI, index: faiss.Index, metadata: Dict, id_to_url: Dict, query: str, top_k: int = 5) -> Dict:
    logging.info("FAISS 검색 함수 호출")
    timings = {}
    try:
        # 쿼리 임베딩 생성
        embed_start = time.perf_counter()
        query_embedding = embed_text(client, [query])[0]
        embed_end = time.perf_counter()
        timings['embedding_creation'] = embed_end - embed_start
        logging.info(f"쿼리 임베딩 생성 시간: {timings['embedding_creation']:.6f}초")

        # FAISS 검색 수행
        search_start = time.perf_counter()
        distances, indices = index.search(np.expand_dims(query_embedding, axis=0), top_k)
        search_end = time.perf_counter()
        timings['faiss_search'] = search_end - search_start
        logging.info(f"FAISS 검색 시간: {timings['faiss_search']:.6f}초")

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in metadata:
                results.append({
                    "distance": dist,
                    "metadata": metadata[idx],
                    "url": id_to_url.get(idx, "Unknown URL")
                })

        return {
            "results": results,
            "closest_distance": distances[0][0] if distances.size > 0 else None,
            "timings": timings
        }
    except Exception as e:
        logging.error(f"FAISS 검색 실패: {str(e)}")
        raise

# BM25 검색 함수
def search_bm25(bm25: BM25Okapi, metadata: Dict, id_to_url: Dict, tokenizer: Okt, query: str, top_k: int = 5) -> Dict:
    logging.info("BM25 검색 함수 호출")
    timings = {}
    try:
        # 질문 토큰화
        token_start = time.perf_counter()
        tokens = tokenizer.morphs(query)
        token_end = time.perf_counter()
        timings['tokenization'] = token_end - token_start
        logging.info(f"쿼리 토큰화 시간: {timings['tokenization']:.6f}초")

        # BM25 점수 계산
        score_start = time.perf_counter()
        scores = bm25.get_scores(tokens)
        score_end = time.perf_counter()
        timings['bm25_score'] = score_end - score_start
        logging.info(f"BM25 점수 계산 시간: {timings['bm25_score']:.6f}초")

        # 상위 인덱스 정렬
        sort_start = time.perf_counter()
        top_indices = np.argsort(scores)[::-1][:top_k]
        sort_end = time.perf_counter()
        timings['sort'] = sort_end - sort_start
        logging.info(f"BM25 상위 인덱스 정렬 시간: {timings['sort']:.6f}초")

        results = []
        for idx in top_indices:
            if idx in metadata:
                results.append({
                    "score": scores[idx],
                    "metadata": metadata[idx],
                    "url": id_to_url.get(idx, "Unknown URL")
                })

        return {
            "results": results,
            "score": scores[top_indices[0]] if len(top_indices) > 0 else None,
            "timings": timings
        }
    except Exception as e:
        logging.error(f"BM25 검색 실패: {str(e)}")
        raise

# BM25 + FAISS 검색 함수
def search_bm25_faiss(client: OpenAI, bm25: BM25Okapi, metadata: Dict, id_to_url: Dict, tokenizer: Okt, query: str, bm25_top_k: int = 10, faiss_top_k: int = 5, dimension: int = 1536) -> Dict:
    timings = {}
    logging.info("BM25 + FAISS 검색 함수 호출")
    try:
        # BM25 검색
        bm25_start = time.perf_counter()
        bm25_results = search_bm25(bm25, metadata, id_to_url, tokenizer, query, top_k=bm25_top_k)
        bm25_end = time.perf_counter()
        timings['bm25_total'] = bm25_end - bm25_start
        logging.info(f"BM25 총 검색 시간: {timings['bm25_total']:.6f}초")

        bm25_docs = bm25_results["results"]

        if not bm25_docs:
            logging.info("BM25 검색 결과가 없습니다.")
            return {
                "results": [],
                "timings": timings
            }

        # BM25 상위 문서들의 텍스트와 테이블을 결합
        combined_texts = []
        for doc in bm25_docs:
            text = doc['metadata'].get("merged_text", "")
            tables = doc['metadata'].get("tables", [])
            tables_text = " ".join([str(table) for table in tables])
            combined_text = text + " " + tables_text
            combined_texts.append(combined_text)

        # BM25 상위 문서들의 임베딩 생성
        embed_start = time.perf_counter()
        doc_embeddings = embed_text(client, combined_texts)
        embed_end = time.perf_counter()
        timings['embedding_creation'] = embed_end - embed_start
        logging.info(f"BM25 상위 문서 임베딩 생성 시간: {timings['embedding_creation']:.6f}초")

        # 쿼리 임베딩 생성
        query_embedding = embed_text(client, [query])[0]
        logging.info("쿼리 임베딩 생성 완료.")

        # FAISS 인덱스 생성 및 검색
        faiss_start = time.perf_counter()
        temp_index = faiss.IndexFlatIP(dimension)  # Inner Product 기반 인덱스
        faiss.normalize_L2(doc_embeddings)
        faiss.normalize_L2(np.expand_dims(query_embedding, axis=0))
        temp_index.add(doc_embeddings)
        distances, indices = temp_index.search(np.expand_dims(query_embedding, axis=0), faiss_top_k)
        faiss_end = time.perf_counter()
        timings['faiss_search'] = faiss_end - faiss_start
        logging.info(f"FAISS 검색 시간: {timings['faiss_search']:.6f}초")

        # FAISS 검색 결과 수집
        faiss_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(bm25_docs):
                doc = bm25_docs[idx]
                faiss_results.append({
                    "distance": dist,
                    "metadata": doc['metadata'],
                    "url": doc['url']
                })

        return {
            "results": faiss_results,
            "timings": timings
        }

    except Exception as e:
        logging.error(f"BM25 + FAISS 검색 실패: {str(e)}")
        raise

# # Gradio 챗봇 함수

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

# Gradio 앱 설정
def main():
    # 초기화 함수 호출
    initialize_logging()
    initialize_search_time_file()

    # OpenAI 클라이언트 초기화
    openai_api_key = "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"  # 환경 변수에서 API 키를 가져옵니다.
    if not openai_api_key:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수 'OPENAI_API_KEY'를 설정해주세요.")
    client = initialize_openai_client(api_key=openai_api_key)

    # FAISS 인덱스 및 메타데이터 로드
    faiss_index_path = "faiss_index.bin"
    metadata_path = "metadata.json"
    index, metadata = load_faiss_index(faiss_index_path, metadata_path)

    # ID <-> URL 매핑 생성
    id_to_url = {idx: data["url"] for idx, data in metadata.items()}
    url_to_id = {data["url"]: idx for idx, data in metadata.items()}

    # BM25 설정
    tokenizer = Okt()
    bm25_corpus_path = "bm25_corpus.pkl"
    bm25_index_path = "bm25_index.pkl"
    bm25 = load_or_create_bm25(bm25_corpus_path, bm25_index_path, metadata, tokenizer)

    # 거리 임계값 설정
    SIMILARITY_THRESHOLD = 0.5

    with gr.Blocks() as demo:
        gr.Markdown("# ISSAC")

        search_method = gr.Radio(choices=['faiss', 'bm25+faiss', 'bm25'], label="검색 방법 선택", value='faiss')

        with gr.Tab("새로운 대화"):
            chat_bot = gr.Chatbot()
            chat_msg = gr.Textbox(placeholder="무엇을 도와드릴까요?", container=False)
            chat_submit = gr.Button("채팅")

            def chat_user_message(user_message, history):
                return "", history + [[user_message, None]]

            def chat_bot_message(history, search_method_selection):
                user_message = history[-1][0]
                bot_message_generator = chat_placeholder(
                    client=client,
                    index=index,
                    metadata=metadata,
                    id_to_url=id_to_url,
                    url_to_id=url_to_id,
                    bm25=bm25,
                    tokenizer=tokenizer,
                    query=user_message,
                    search_method=search_method_selection
                )
                bot_message = ""
                for chunk in bot_message_generator:
                    bot_message += chunk
                    history[-1][1] = bot_message
                    yield history
                

            chat_msg.submit(chat_user_message, [chat_msg, chat_bot], [chat_msg, chat_bot], queue=False).then(
                chat_bot_message, [chat_bot, search_method], chat_bot
            )
            chat_submit.click(chat_user_message, [chat_msg, chat_bot], [chat_msg, chat_bot], queue=False).then(
                chat_bot_message, [chat_bot, search_method], chat_bot
            )

        examples = [
            "셔틀버스 운행경로",
            "VSCODE 연결이 끊기는 문제를 어떻게 해결할 수 있나요?",
            "외국어학당 수강 신청 기간",
            "What should I do if my job is stuck in pending status?",
        ]

        def set_example(example):
            return gr.Textbox.update(value=example)

        gr.Examples(
            examples=examples,
            inputs=[chat_msg],
            outputs=[chat_msg],
            fn=set_example,
            cache_examples=False,
        )

    demo.launch(server_name="0.0.0.0", server_port=7860,share=True)

if __name__ == "__main__":
    main()