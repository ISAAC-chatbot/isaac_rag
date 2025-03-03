import gradio as gr
import hashlib
from utils.logging_utils import (
    log_query,
    get_logger_for_user)
from fastapi import FastAPI
from gradio import mount_gradio_app
from utils.data_loader import initialize_client
from utils.response import (generate_response, detect_language)
from utils.conversation import ConversationManager
from fastapi.responses import RedirectResponse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from custom_js_content import custom_js
from custom_css_content import custom_css
from api import router

import logging
import re
import html


app = FastAPI()

# 세션별 ConversationManager 저장
session_managers = {}


def main():
    
    client = initialize_client()

    # FAISS 인덱스 및 메타데이터 로드
    # index, metadata = load_faiss_index()

    # ID <-> URL 매핑 생성
    # id_to_url = {idx: data["url"] for idx, data in metadata.items()}
    # url_to_id = {data["url"]: idx for idx, data in metadata.items()}

    # BM25 설정
    # tokenizer = Okt()
    # bm25 = load_or_create_bm25(metadata, tokenizer)

 
    

    def generate_session_id(request: gr.Request):
        host = request.client.host
        ip_hash = hashlib.sha256(host.encode()).hexdigest()
        del host
        session_hash = request.session_hash
        user_agent = request.headers["user-agent"]
        # session_id = f"{session_hash}-{user_agent}"
        session_id = f"{session_hash}-ip hash:{ip_hash}-{user_agent}"
        return session_id
    


    with gr.Blocks(
        head= custom_js,
        css= custom_css,
        theme=gr.themes.Default(
            font=["Pretendard", "system-ui", "sans-serif"],
            font_mono=["Pretendard", "monospace"]
        )
    ) as demo:

        # 제목 및 설명
        gr.Markdown("# ISAAC")
        gr.Markdown(
            "연세대학교에 대해 궁금한 게 있다면 뭐든 물어보세요!<br>"
            "더 구체적으로 질문하실수록 더욱 정확한 답변을 드릴 수 있어요! :)",
            elem_id="info-link"
        )

        # 검색 방법 선택 라디오 버튼
        search_method = gr.Radio(
            choices=[
                'ISAAC Basic - 균형 잡힌 검색',
                'ISAAC Pro - 자세한 검색'
            ],
            label="검색 방법 선택",
            value='ISAAC Basic - 균형 잡힌 검색'
        )
        

        with gr.Tab("새로운 대화"):
            chat_bot = gr.Chatbot(elem_id="chatbot")
            chat_msg = gr.Textbox(placeholder="무엇을 도와드릴까요?", container=False, elem_id="chat-msg")
            chat_submit = gr.Button("채팅", elem_id="chat-submit")


            def chat_user_message(user_message, history, *args):
                history = history + [[user_message, None]]
                return "", history

            def chat_bot_message(history, search_method_selection, request: gr.Request):

                issac_session = generate_session_id(request)
                session_id=  hashlib.sha256(issac_session.encode()).hexdigest()
                logger = get_logger_for_user(session_id) or logging.getLogger()
                user_message = history[-1][0]
                logger.info(f"[세션 {issac_session}] \n 사용자 쿼리: {user_message}")

                # Log the query
                log_query(session_id, user_message, search_method_selection)
                logger.info(f"선택된 검색 방법: {search_method_selection}")

                # 세션별 ConversationManager 가져오기 또는 생성
                if session_id not in session_managers:
                    session_managers[session_id] = ConversationManager(
                        client=client,
                        max_queries=5,
                        search_config={
                            "search_method": search_method_selection,
                            "top_k" : 5
                        },
                        logger = logger
                    ) 
                else:
                    # 기존 ConversationManager의 검색 설정 업데이트
                    session_managers[session_id].update_search_config({
                        "search_method": search_method_selection
                    })
                
                conversation_manager = session_managers[session_id]
                
                # 언어 감지
                language = detect_language(user_message, logger)
                
                # 로그 기록
                logger.info(f"새로운 쿼리 수신: {user_message} (언어: {language})")

                # 상태 그래프를 통해 메시지 처리 (쿼리 재작성 및 문서 검색)
                final_state = conversation_manager.process_message(user_message, language=language)

                # 응답 생성이 필요한지 확인
                if final_state.get("needs_response"):
                    rewritten_query = final_state.get("rewritten_query", user_message)
                    context = final_state.get("context", [])

                    # 응답 생성 (스트리밍)
                    bot_message_generator = generate_response(
                        client=client,
                        rewritten_query=rewritten_query,
                        original_query=user_message,
                        context = context,
                        conversation_manager=conversation_manager,
                        language = language,
                        logger = logger
                    )

                    bot_message = ""
                    url_message = ""
                    first_flag = False
                    second_flag = False
                    is_source_section = False
                    buffer = ""
                    url_pattern = re.compile(r"https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+")
                    for chunk in bot_message_generator:
                        buffer += chunk

                        if chunk == "출" :
                            first_flag = True
                            continue

                        if first_flag and chunk == "처":
                            second_flag = True
                            first_flag = False
                            continue
                        elif first_flag and chunk != "처":
                            bot_message += "출" + chunk
                            first_flag = False

                        if second_flag and chunk == ":":
                            is_source_section = True
                            first_flag = False
                            second_flag = False
                            continue
                        elif first_flag and second_flag and chunk != ":":
                            bot_message += "출처" + chunk
                            first_flag = False
                            second_flag = False

                        # Check for "출처:" or "출처 :"
                        if "출처:" in buffer or "출처 :" in buffer:
                            is_source_section = True  # Start collecting URL
                            buffer = ""  # Clear buffer after detecting "출처"
                            continue

                        if is_source_section:
                            # Collect URL content after "출처:"
                            url_message += chunk.strip()  # Add chunk to URL and strip whitespace
                            continue

                        # If not in the "출처:" section, add chunk to the response
                        if (not first_flag) and (not second_flag) and (not is_source_section):
                            bot_message += chunk

                        # Update history and yield response
                        history[-1][1] = bot_message
                        yield history

                    del buffer

                    # URL 추출 및 클린업
                    match = url_pattern.search(url_message)
                    if match:
                        raw_url = match.group()
                        # URL에서 실제 링크만 추출하는 정규식
                        url_only_pattern = re.compile(r'https?://[^\s\]\)]+')
                        url_matches = url_only_pattern.findall(url_message)
                        
                        if url_matches:
                            # 중복된 URL 제거하고 첫 번째 URL만 사용
                            clean_url = url_matches[0]
                            # URL 끝의 특수문자 제거
                            clean_url = re.sub(r'[)\]}>]+$', '', clean_url)
                        else:
                            clean_url = raw_url
                    else:
                        clean_url = "https://www.yonsei.ac.kr/sc/"
                    
                    # URL 디코딩 및 안전한 HTML 출력
                    clean_url = html.unescape(clean_url)
                    clean_url_escape = html.escape(clean_url)

                    history[-1][1] = bot_message + "<p></p>" + f"""
                    <div class="feedback-container">
                        <div class="source-button-container">
                            <button class="icon-button source-button">
                                <a href="{clean_url_escape}" target="_blank" style="color: inherit; text-decoration: none;">
                                    🔗 출처
                                </a>
                            </button>
                        </div>
                        <div class="feedback-buttons">
                            <button class="icon-button like">좋아요👍</button>
                            <button class="icon-button dislike">싫어요👎</button>
                        </div>
                    </div>
                    """
                    yield history
                        

                    # 응답 생성 완료 후 대화 기록 업데이트
                    conversation_manager.update_query_history(user_message, rewritten_query, bot_message)
                else:
                    # 요약 등 다른 처리가 필요한 경우
                    bot_response = final_state.get("response", "죄송합니다. 응답을 생성할 수 없습니다.")
                    history[-1][1] = bot_response + """
                    <div class="feedback-buttons">
                        <button class="icon-button like">좋아요👍</button>
                        <button class="icon-button dislike">싫어요👎</button>
                    </div>
                    """
                    yield history

            search_method.change(
                fn=lambda selected: selected, 
                inputs=search_method,
                outputs=None,
                js="(selected_value) => { trackSearchMethodChange(selected_value); }"
            )


            # 채팅 메시지 제출 이벤트
            chat_msg.submit(
                chat_user_message,
                inputs=[chat_msg, chat_bot],
                outputs=[chat_msg, chat_bot],
                queue=False,
                js="(message, history) => { trackChatMessage(message); return [message, history]; }"
            ).then(
                chat_bot_message,
                inputs=[chat_bot, search_method],
                outputs=[chat_bot]
            )
            
            chat_submit.click(
                chat_user_message,
                inputs=[chat_msg, chat_bot],
                outputs=[chat_msg, chat_bot],
                queue=False,
                js="(message, history) => { trackChatMessage(message); return [message, history]; }"
            ).then(
                chat_bot_message,
                inputs=[chat_bot, search_method],
                outputs=[chat_bot]
            )
        
        # 예제 질문들
        examples = [
            "2025-1 학부 등록금 납부 일정",
            "2025-1학기 추가등록 언제야?",
            "2025-1학기 수강신청 일정",
            "2025-1학기 S/U 제도에 대해 알려줘",
            "2025-1학기 수강과목 철회 중간고사 이전이야?",
            "마일리지 총량에 대해서 과별로 정리해줘",
            "2025-1 송도학사 입사 일정"
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
        
        # 사용자 피드백 박스
        gr.Markdown("""
        <div id="feedback-box">
            <b>ISAAC 사용 경험을 남겨주세요!</b><br>
            추첨을 통해 스타벅스 기프티콘을 드립니다.<br>
            <a id="feedback-link" href="https://forms.gle/UMqiUYzJik9xf1YX6" target="_blank">피드백 남기기</a>
        </div>
        """, elem_id="feedback-box")
    
    return demo

if __name__ == "__main__":
    # demo = main()
    # demo.launch(server_name="0.0.0.0", server_port=8088, share=True,
    #     show_error=True,  # 에러 상세 표시
    #     show_api=False,   # API 엔드포인트 비활성화
    #     favicon_path=None # 기본 파비콘 사용)
    # )

    # API 모듈 포함 (Gradio와 동일한 포트에서 실행)
    

    @app.get("/android")
    def download_android_apk():
        """
        S3에 저장된 APK 파일로 리디렉션
        """
        return RedirectResponse(url="https://issac-dev.s3.ap-northeast-2.amazonaws.com/ISAAC-release.apk")

    app.include_router(router, prefix="/api")
    
    # Gradio 앱 실행
    gradio_app = main()
    app = mount_gradio_app(app, gradio_app, path="/")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)
    