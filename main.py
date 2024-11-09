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
    ga_script = """
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-5YRFXGSK7Y"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', 'G-5YRFXGSK7Y');
    
    
    /*페이지 로드 시 타이머 시작*/
    let sessionStartTime = Date.now();

    /*페이지를 떠날 때 세션 시간 기록*/
    window.addEventListener('beforeunload', function() {
        let sessionEndTime = Date.now();
        let sessionDuration = (sessionEndTime - sessionStartTime) / 1000; // 초 단위

        gtag('event', 'session_duration', {
            'event_category': 'User Session',
            'event_label': 'Session Time on Search Method Page',
            'value': sessionDuration
        });
    });
    
    // search_method 선택 시 이탈 타이머 초기화
    function trackSearchMethodChange(selectedSearchMethod) {
        gtag('event', 'search_method_change', {
            'event_category': 'Search Method Selection',
            'event_label': selectedSearchMethod,
            'value': 1
        });

        // 일정 시간 후 추가 활동 없으면 이탈로 간주
        setTimeout(function() {
            gtag('event', 'bounce', {
                'event_category': 'User Engagement',
                'event_label': 'No further activity after search method selection',
                'value': 1
            });
        }, 30000); // 30초 동안 추가 활동이 없으면 이탈로 간주
    }

    /*기기 유형*/
    const isMobile = /Mobi|Android/i.test(navigator.userAgent);
    const deviceType = isMobile ? 'mobile' : 'desktop';
    gtag('event', 'device_type_record', {
    'device_type': deviceType
    });

    /*도움 버튼*/ 
    document.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('icon-button') && e.target.classList.contains('like')) {
            gtag('event', 'like_click', {
                'event_category': 'User Feedback',
                'event_label': '👍 도움이 됐어요',
                'value': 1
            });
        } else if (e.target && e.target.classList.contains('icon-button') && e.target.classList.contains('dislike')) {
            gtag('event', 'dislike_click', {
                'event_category': 'User Feedback',
                'event_label': '👎 별로예요',
                'value': 1
            });
        }
    });
    
    function trackSearchMethodChange(selectedSearchMethod) {
        let shortenedLabel;
        if (selectedSearchMethod.includes("ISAAC 2.0 - 정확하고 자세한 정보검색")) {
            shortenedLabel = "ISAAC 2.0";
        } else if (selectedSearchMethod.includes("ISAAC 2.0-turbo : 하이브리드형 검색")) {
            shortenedLabel = "ISAAC 2.0-turbo";
        } else if (selectedSearchMethod.includes("ISAAC Lite : 가볍고 빠른 검색")) {
            shortenedLabel = "ISAAC Lite";
        }

        gtag('event', 'search_method_change', {
            'event_category': 'Search Method Selection',
            'event_label': shortenedLabel,
            'value': 1
        });
    }
    </script>
    """
    
    with gr.Blocks(head=ga_script, css="""
    #feedback-box {
        background-color: #fffbf2;
        padding: 15px;
        border-radius: 8px;
        font-size: 16px;
        color: #3d3d3d;
        text-align: center;
        margin-top: 20px;
        line-height: 1.5;
        border: 1px solid #e0d7c3;
    }

    #feedback-box b {
        font-weight: bold;
        color: #000000;
    }

    #feedback-link {
        color: #0073e6;
        text-decoration: underline;
    }

    /* 피드백 버튼 스타일 : 좋아요 싫어요 */
    .feedback-buttons {
        display: flex;
        justify-content: flex-start;
        margin-top: 5px;
    }

    .icon-button {
        cursor: pointer;
        padding: 5px 10px;
        border: none;
        background-color: transparent;
        font-size: 16px;
    }

    .icon-button.like:hover {
        color: #A9A9A9;
    }

    .icon-button.dislike:hover {
        color: #A9A9A9;
    }
    """) as demo:
        gr.Markdown("# ISAAC")
        gr.Markdown("연세대학교의 모든 정보에 대해 질문해주세요!<br>\"~에 대해 알려줘\"라고 입력하시면 더 정확한 정보를 받아보실 수 있습니다 :)")

        # 검색 방법 선택
        search_method = gr.Radio(
            choices=['ISAAC 2.0 - 정확하고 자세한 정보검색', 'ISAAC 2.0-turbo : 하이브리드형 검색', 'ISAAC Lite : 가볍고 빠른 검색'], label="검색 방법 선택", value='ISAAC 2.0 - 정확하고 자세한 정보검색'
        )
        
        with gr.Tab("새로운 대화"):
            chat_bot = gr.Chatbot()
            chat_msg = gr.Textbox(placeholder="무엇을 도와드릴까요?", container=False)
            chat_submit = gr.Button("채팅")

            def chat_user_message(user_message, history):
                history = history + [[user_message, None]]
                return "", history

            def chat_bot_message(history, search_method_selection):
                user_message = history[-1][0]
                
                # bot_message_generator에서 텍스트를 생성하는 부분
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
                    
                combined_message = bot_message + """
                    <div class="feedback-buttons" style="display: flex; justify-content: flex-start; gap: 8px; margin-top: 10px;">
                        <button class="icon-button like">👍 도움이 됐어요</button>
                        <button class="icon-button dislike">👎 별로예요</button>
                    </div>
                """
                
                bot_message_with_feedback = combined_message
                
                history[-1][1] = bot_message_with_feedback 

                return history
            
            search_method.change(
                fn=lambda selected: selected,  # 선택된 검색 방법 값을 유지
                inputs=search_method,
                outputs=None,
                js="(selected_value) => { trackSearchMethodChange(selected_value); }"
            )
            
            # 이벤트 핸들러 수정 : 비동기적인 대화 처리를 위해서 
            chat_msg.submit(
                chat_user_message,
                inputs=[chat_msg, chat_bot],
                outputs=[chat_msg, chat_bot],
                queue=False
            ).then(
                chat_bot_message,
                inputs=[chat_bot, search_method],
                outputs=[chat_bot]
            )
            chat_submit.click(
                chat_user_message,
                inputs=[chat_msg, chat_bot],
                outputs=[chat_msg, chat_bot],
                queue=False
            ).then(
                chat_bot_message,
                inputs=[chat_bot, search_method],
                outputs=[chat_bot]
            )
        
        examples = [
            "신촌캠퍼스 셔틀버스 운행 시간 알려줘.",
            "1대 총장님 성함이 뭐야?",
            "2024 2학기 수강신청 일정 알려줘.",
            "부여받는 최대 마일리지가 몇이야?",
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
        
        gr.Markdown("""
        <div id="feedback-box">
            <b>ISAAC 사용 경험을 남겨주세요!</b><br>
            추첨을 통해 스타벅스 기프티콘을 드립니다.<br>
            <a id="feedback-link" href="https://docs.google.com/forms/d/1YL6f23nWhXGH8lOasOXnvnZREeVNBN30lOffzt2fc84/edit" target="_blank">피드백 남기기</a>
        </div>
        """, elem_id="info-link")

    # Gradio 인터페이스 실행
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()