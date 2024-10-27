# main.py

import gradio as gr
from utils.logging_utils import initialize_logging, initialize_search_time_file
from utils.config import OPENAI_API_KEY, LOG_FILE, FAISS_INDEX_PATH, METADATA_PATH, BM25_INDEX_PATH, BM25_CORPUS_PATH
from utils.data_loader import initialize_openai_client, load_faiss_index, load_or_create_bm25
from utils.search import search_faiss, search_bm25, search_bm25_faiss
from utils.response import generate_response_with_context, detect_language, chat_placeholder
from utils.conversation import ConversationManager
from utils.embeddings import embed_text
from konlpy.tag import Okt
import os

def main():
    # 초기화 함수 호출
    initialize_logging(log_file=LOG_FILE)
    initialize_search_time_file()

    # OpenAI 클라이언트 초기화
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수 'OPENAI_API_KEY'를 설정해주세요.")
    client = initialize_openai_client(api_key=OPENAI_API_KEY)

    # FAISS 인덱스 및 메타데이터 로드
    index, metadata = load_faiss_index()

    # ID <-> URL 매핑 생성
    id_to_url = {idx: data["url"] for idx, data in metadata.items()}
    url_to_id = {data["url"]: idx for idx, data in metadata.items()}

    # BM25 설정
    tokenizer = Okt()
    bm25 = load_or_create_bm25(metadata, tokenizer)

    # Gradio 앱 설정 및 실행
    with gr.Blocks() as demo:
        gr.Markdown("# ISSAC")

        search_method = gr.Radio(choices=['faiss', 'bm25+faiss', 'bm25'], label="검색 방법 선택", value='faiss')

        with gr.Tab("새로운 대화"):
            chat_bot = gr.Chatbot()
            chat_msg = gr.Textbox(placeholder="무엇을 도와드릴까요?", container=False)
            chat_submit = gr.Button("채팅")

            conversation_manager = ConversationManager(max_length=5)

            def chat_user_message(user_message, history):
                return "", history + [[user_message, None]]

            def chat_bot_message(history, search_method_selection):
                user_message = history[-1][0]

                # 대화 매니저를 통해 맥락 업데이트
                conversation_manager.add_message(client, user_message)

                # 현재 맥락 가져오기
                context_text = conversation_manager.get_context()

                # 검색 및 응답 생성 시 맥락 사용
                bot_message_generator = chat_placeholder(
                    client=client,
                    index=index,
                    metadata=metadata,
                    id_to_url=id_to_url,
                    url_to_id=url_to_id,
                    bm25=bm25,
                    tokenizer=tokenizer,
                    query=context_text,  # 수정된 부분: 맥락 전체를 쿼리로 사용
                    search_method=search_method_selection
                )
                bot_message = ""
                for chunk in bot_message_generator:
                    bot_message += chunk
                    history[-1][1] = bot_message
                    yield history

                # 어시스턴트의 응답이 완료된 후 응답 시간 업데이트
                conversation_manager.update_response_time()

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

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()
