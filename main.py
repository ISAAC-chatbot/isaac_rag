import gradio as gr
import hashlib
from utils.logging_utils import (
    initialize_logging, 
    # initialize_search_time_file, 
    initialize_query_logging, 
    log_query,
    get_logger_for_user)

from utils.data_loader import initialize_openai_client
from utils.response import (generate_response, detect_language)
from utils.conversation import ConversationManager
import logging
import uuid
import os
import re
import html

from dotenv import load_dotenv
# .env 파일을 로드합니다
load_dotenv('/home/ubuntu/multiturn/.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)

# 세션별 ConversationManager 저장
session_managers = {}

def main():
    # 초기화 함수 호출
    initialize_logging(log_file='logs/ai_global.log')
    # initialize_search_time_file()
    initialize_query_logging()

    # OpenAI 클라이언트 초기화
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수 'OPENAI_API_KEY'를 설정해주세요.")

        if not OPENAI_API_KEY.startswith('sk-'):
            raise ValueError("잘못된 API 키 형식입니다. OpenAI API 키는 'sk-'로 시작해야 합니다.")

        client = initialize_openai_client(api_key=OPENAI_API_KEY)

    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        raise

    # FAISS 인덱스 및 메타데이터 로드
    # index, metadata = load_faiss_index()

    # ID <-> URL 매핑 생성
    # id_to_url = {idx: data["url"] for idx, data in metadata.items()}
    # url_to_id = {data["url"]: idx for idx, data in metadata.items()}

    # BM25 설정
    # tokenizer = Okt()
    # bm25 = load_or_create_bm25(metadata, tokenizer)

    # Google Analytics 및 좋아요/싫어요 버튼 클릭 로직
    custom_js = r"""

    <!-- Pretendard 폰트 설정 -->
    <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard-dynamic-subset.min.css" />

    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-23DWCGG61N"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag() { dataLayer.push(arguments); }

    // Gradio 푸터 숨기기 (MutationObserver 사용)
    const observer = new MutationObserver((mutations) => {
        const footer = document.querySelector('footer.svelte-sar7eh');
        if (footer) {
            footer.style.display = 'none';
            observer.disconnect(); // 푸터를 찾아서 숨기면 관찰 중단
        }
    });

    // body 요소의 변경 감시 시작
    observer.observe(document.body, {
        childList: true, // 자식 요소 추가/제거 감시
        subtree: true    // 하위 요소들도 감시
    });

    // 기본 동의 설정 (모든 스토리지 거부)
    gtag('consent', 'default', {
        'ad_storage': 'denied',
        'analytics_storage': 'denied',
        'ad_user_data': 'denied',
        'ad_personalization': 'denied'
    });

    // 쿠키 동의 배너 HTML
    const consentBanner = `
        <div id="cookie-consent-banner">
            <span>이 웹사이트는 사용자 경험 개선을 위해 쿠키를 사용합니다.</span>
            <button id="accept-cookies">동의</button>
        </div>
    `;

    // 쿠키 동의 상태 확인 및 설정
    function checkCookieConsent() {
        const consent = localStorage.getItem('cookie-consent');
        if (consent == 'denied' || consent == null) {
            document.body.insertAdjacentHTML('beforeend', consentBanner);
            
            document.getElementById('accept-cookies').addEventListener('click', function() {
                // 동의 상태 업데이트
                gtag('consent', 'update', {
                    'ad_storage': 'granted',
                    'analytics_storage': 'granted',
                    'ad_user_data': 'granted',
                    'ad_personalization': 'granted'
                });
                
                // 동의 상태 저장
                localStorage.setItem('cookie-consent', 'granted');
                
                // 배너 제거
                document.getElementById('cookie-consent-banner').remove();
                
                console.debug('[Consent Granted]', {
                    timestamp: new Date().toISOString(),
                    userId: userId
                });
            });
        } else {
            // 이미 동의한 경우
            gtag('consent', 'update', {
                'ad_storage': 'granted',
                'analytics_storage': 'granted',
                'ad_user_data': 'granted',
                'ad_personalization': 'granted'
            });
        }
    }

    // Google Analytics 초기화 (동의 설정 후)
    gtag('js', new Date());
    gtag('config', 'G-23DWCGG61N', {
        'debug_mode': true
    });

    // DOM 로드 시 쿠키 동의 확인
    document.addEventListener('DOMContentLoaded', checkCookieConsent);

    
    // 사용자 세그먼트 분석을 위한 변수들
    let visitCount = parseInt(localStorage.getItem('visit_count') || '0');
    const entryPath = document.referrer;
    const utmParams = new URLSearchParams(window.location.search);

    // 기존 변수들을 window 객체에 바인딩
    window.chatMetrics = {
        sessionStartTime: Date.now(),
        messageCount: 0,
        methodChangeCount: 0,
        lastMessageTime: Date.now(),
        lastAction: '',
        followUpMessage: false
    };

    // 초기화 시 세션 ID 확인
    function initializeSession() {
        const userId = localStorage.getItem('session_id') || generateUserId();
        console.debug('Session initialized with ID:', userId);
        return userId;
    }

    // 고유 사용자 ID 생성
    function generateUserId() {
            const existingId = localStorage.getItem('session_id');
            if (existingId) {
                return existingId;
            }

            // 새로운 세션 ID 생성
            const timestamp = new Date().getTime();
            const randomStr = Math.random().toString(36).substr(2, 9);
            const id = `session_${timestamp}_${randomStr}`;
            
            // localStorage에 세션 ID 저장
            localStorage.setItem('session_id', id);
            
            // 방문 기록 저장
            const visits = JSON.parse(localStorage.getItem('visit_history') || '[]');
            visits.push({
                sessionId: id,
                timestamp: timestamp,
                userAgent: navigator.userAgent
            });
            localStorage.setItem('visit_history', JSON.stringify(visits));
            
            return id;
    }

    
    // 초기화 호출
    const userId = initializeSession();


    // 즉시 실행 함수로 초기화 코드 래핑
    (function initializeTracking() {
        // 초기화 함수 정의
        function initializeAnalytics() {
            checkCookieConsent();
            trackUserSegment();
            trackPageFlow('entry', {
                'referrer': document.referrer,
                'landing_page': window.location.href,
                'consent_status': 'previously_granted'
            });
        }

        // DOMContentLoaded 이벤트에 바인딩
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeAnalytics);
        } else {
            // 이미 DOM이 로드된 경우 즉시 실행
            initializeAnalytics();
        }
    })();

    // 콘솔 로그 함수 수정
    function logChatMetrics() {
        console.debug('Chat Metrics:', {
            sessionStartTime: window.chatMetrics.sessionStartTime,
            messageCount: window.chatMetrics.messageCount,
            methodChangeCount: window.chatMetrics.methodChangeCount,
            lastMessageTime: window.chatMetrics.lastMessageTime,
            lastAction: window.chatMetrics.lastAction,
            followUpMessage: window.chatMetrics.followUpMessage
        });
    }

    // ============================== 주요 이벤트 전송 ======================================

    // 주요 이벤트 전송 함수 (기존 함수 확장)
    function sendGAEvent(eventName, eventParams) {
        try {
            if (typeof gtag === 'undefined') {
                console.warn('[GA Warning] gtag is not loaded. Event not sent:', eventName);
                return;
            }

            const cleanParams = {
                ...eventParams,
                client_timestamp: new Date().toISOString(),
                user_id: userId,
                visit_count: visitCount
            };

            gtag('event', eventName, cleanParams);
            console.debug(`[GA Event Sent] ${eventName}:`, cleanParams);
        } catch (error) {
            console.debug(`[GA Event Error] ${eventName}:`, error);
        }
    }

    // ============================== 메시지 분석 ======================================

    // 메시지 카운트 증가를 위한 단일 함수
    function incrementMessageCount() {
        window.chatMetrics.messageCount++;
        return window.chatMetrics.messageCount;
    }

    // 채팅 메시지 전송 추적
    function trackChatMessage(message) {
        const currentCount = incrementMessageCount();
        window.chatMetrics.lastMessageTime = Date.now();
        window.chatMetrics.followUpMessage = true;
        window.chatMetrics.lastAction = 'user_message';

        sendGAEvent('chat_message_sent', {
            'event_category': 'Chat Interaction',
            'message_length': message.length,
            'message_count': currentCount,
        });
    }

    // updateChatMetrics 함수 수정
    function updateChatMetrics(eventType) {
        const metrics = window.chatMetrics;
        
        switch(eventType) {
            case 'message':
                // messageCount는 trackChatMessage에서만 증가
                metrics.lastMessageTime = Date.now();
                metrics.lastAction = 'user_message';
                metrics.followUpMessage = true;
                break;
            case 'method_change':
                metrics.methodChangeCount++;
                metrics.lastAction = 'search_method_change';
                break;
        }
        
        logChatMetrics();
    }

    // Gradio 컴포넌트에 이벤트 리스너 연결
    document.addEventListener('DOMContentLoaded', function() {
        // 메시지 전송 버튼 클릭 시
        const submitButton = document.querySelector('.chat-submit');
        if (submitButton) {
            submitButton.addEventListener('click', () => updateChatMetrics('message'));
        }

        // 검색 방법 변경 시
        const searchMethodSelect = document.querySelector('select[data-testid="select"]');
        if (searchMethodSelect) {
            searchMethodSelect.addEventListener('change', () => updateChatMetrics('method_change'));
        }
    });

    // ============================== 검색 방법 변경 추적 ======================================

    function trackSearchMethodChange(selectedMethod) {
        updateChatMetrics('method_change');
        sendGAEvent('search_method_change', {
            'event_category': 'Search Method',
            'event_label': selectedMethod,
            'method_change_count': window.chatMetrics.methodChangeCount
        });
    }

    // ============================== 사용자 세그먼트 추적 ======================================


    // 사용자 세그먼트 추적
    function trackUserSegment() {
        visitCount++;
        localStorage.setItem('visit_count', visitCount);
        localStorage.setItem('last_visit', new Date().toISOString());

        sendGAEvent('user_segment', {
            'event_category': 'User Info',
            'user_id': userId,
            'visit_count': visitCount,
            'visit_type': visitCount === 1 ? 'new' : 'returning',
            'utm_source': utmParams.get('utm_source'),
            'isMobile': /iPhone|iPad|iPod|Android/i.test(navigator.userAgent),
            'deviceType': /iPad|Tablet/i.test(navigator.userAgent) ? 'tablet' : (/iPhone|Android/i.test(navigator.userAgent) ? 'mobile' : 'desktop'),
            'platform': navigator.platform,
            'screenSize': `${window.screen.width}x${window.screen.height}`,
            'browser': navigator.userAgent.match(/(opera|chrome|safari|firefox|msie|trident(?=\/))\/?\s*(\d+)/i)[1],
            'language': navigator.language || navigator.userLanguage,
            'orientation': window.screen.orientation ? window.screen.orientation.type : 'undefined',
            'viewport width':window.innerWidth,
            'viewport height': window.innerHeight
                
        });
    }

    // ============================== 버튼 클릭 이벤트 추적 ======================================

    // 피드백 버튼 클릭 이벤트 추적
    document.addEventListener('click', function (e) {
        if (e.target && e.target.classList.contains('icon-button')) {
            if (e.target.classList.contains('like')) {
                // 좋아요 버튼 클릭 시
                const dislikeButton = e.target.closest('.feedback-buttons').querySelector('.dislike');
                e.target.classList.add('clicked');
                e.target.disabled = true;
                e.target.style.cursor = 'not-allowed';
                dislikeButton.disabled = true;
                dislikeButton.style.opacity = '0.5';
                dislikeButton.style.cursor = 'not-allowed';
                
                sendGAEvent('feedback_click', {
                    'event_category': 'User Feedback',
                    'feedback_type': 'positive'
                });
            } else if (e.target.classList.contains('dislike')) {
                // 싫어요 버튼 클릭 시
                const likeButton = e.target.closest('.feedback-buttons').querySelector('.like');
                e.target.classList.add('clicked');
                e.target.disabled = true;
                e.target.style.cursor = 'not-allowed';
                likeButton.disabled = true;
                likeButton.style.opacity = '0.5';
                likeButton.style.cursor = 'not-allowed';
                
                sendGAEvent('feedback_click', {
                    'event_category': 'User Feedback',
                    'feedback_type': 'negative'
                });
            }
        }
    });

    // 출처 클릭 이벤트 추적
    document.addEventListener('click', function (e) {
        if (e.target && (e.target.classList.contains('source-button') || e.target.closest('.source-button'))) {
            const button = e.target.closest('.source-button');
            const link = button.querySelector('a');
            const url = link.getAttribute('href');
            sendGAEvent('source_click', {
                'event_category': 'Source Reference',
                'url': url,
            });
        }
    });

    // ============================== 페이지 흐름 추적 ======================================

    // 페이지 흐름 추적
    function trackPageFlow(eventType, details = {}) {
        sendGAEvent(eventType, {
            'event_category': 'User Navigation',
            'page_path': window.location.pathname,
            'entry_path': entryPath,
            'time_on_page': Math.floor((Date.now() - window.chatMetrics.sessionStartTime) / 1000),
            'user_id': userId,
            'visit_count': visitCount,
            ...details
        });
    }

    // ============================== 세션 종료 추적 ======================================

    // 세션 종료 추적
    window.addEventListener('beforeunload', function() {
        trackSessionEnd('exit');  // 페이지 이탈
    });

    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'hidden') {
            trackSessionEnd('visibility_change');  // 탭 전환/최소화
        }
    });

    // 세션 종료 추적 함수
    function trackSessionEnd(exitType) {
        const sessionDuration = (Date.now() - window.chatMetrics.sessionStartTime) / 1000;
        const avgMessages = window.chatMetrics.messageCount / sessionDuration;
        const dislikeClicked = document.querySelector('.dislike')?.classList.contains('clicked');

        sendGAEvent(exitType, {
            'event_category': 'User Session',
            'session_duration': sessionDuration,
            'message_count': window.chatMetrics.messageCount,
            'avg_messages_per_minute': avgMessages,
            'last_action': window.chatMetrics.lastAction,
            'dislike_before_exit': dislikeClicked,
            'visit_count': visitCount,
            'is_returning': visitCount > 1,
            'exit_type': exitType  // 명시적으로 종료 유형 전달
        });

        // 페이지 이탈 추적 (실제 이탈인 경우만)
        if (exitType === 'exit') {
            trackPageFlow('exit', {
                'exit_page': window.location.pathname,
                'session_duration': sessionDuration,
                'interaction_count': window.chatMetrics.messageCount,
                'dislike_clicked': dislikeClicked
            });
        }
    }

    // 싫어요 버튼 클릭 상태 추적을 위한 코드 추가
    document.addEventListener('click', function (e) {
        if (e.target && e.target.classList.contains('dislike')) {
            e.target.classList.add('clicked');  // 클릭 상태 표시
            window.chatMetrics.lastAction = 'dislike_feedback';
        }
    });

    </script>

    """

    # CSS 스타일링 업데이트
    custom_css = """

/* 폰트 설정 */
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}

/* 시스템 폰트 대체 설정 */
body {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, "Helvetica Neue", "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
}

/* 강제 색상 모드 지원 */
@media (forced-colors: active) {
    .icon-button {
        background-color: ButtonFace;
        color: ButtonText;
        border-color: ButtonBorder;
    }

    .icon-button:hover {
        background-color: ButtonFace;
        color: ButtonText;
    }

    .source-button-container .icon-button,
    .icon-button.like,
    .icon-button.dislike {
        background-color: ButtonFace;
        color: ButtonText;
    }
}

/* 상태 의사 클래스 표준 구문 사용 */
:state(hover) {
    background-color: var(--hover-color);
}

:state(active) {
    background-color: var(--active-color);
}

:state(focus) {
    outline: 2px solid var(--focus-color);
    outline-offset: 2px;
}

/* 전체적인 박스 사이징 설정 */
* {
    box-sizing: border-box;
}

/* 전체 배경 */
body {
    background-color: rgba(216, 216, 214, 0.15); /* 불투명한 배경색 */
}

/* Gradio 전체 컨테이너 */
.gradio-container {
    padding: 20px;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue', 'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}

/* 제목 텍스트 */
h1, h2, h3, #info-link {
    color: #ffa976;
    font-weight: 700;
}

/* 채팅박스 스타일링 */
.gradio-chat-message .bot, .gradio-chat-message .user {
    border-radius: 16px;
    margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(187, 175, 189, 0.1);
}

/* feedback-box 영역 스타일 */
#feedback-box {
    background-color: rgba(249, 217, 192, 0.15); /* 불투명한 배경색 */
    padding: 20px;
    border-radius: 12px;
    font-size: 16px;
    color: #444;
    text-align: center;
    margin-top: 25px;
    line-height: 1.6;
    border: 1px solid #e8d3ac;
}

#feedback-box b {
    font-weight: 600;
    color: #66b2b2;
}

#feedback-link {
    color: #66b2b2;
    text-decoration: none;
    font-weight: 600;
    transition: all 0.3s ease;
}

#feedback-link:hover {
    color: #559999;
    text-decoration: underline;
}

/* 피드백 버튼 컨테이너 */
.feedback-container {
    display: flex;
    justify-content: space-between; /* 좌우 정렬 */
    align-items: center; /* 수직 중앙 정렬 */
    margin-top: 18px;
    gap: 12px; /* 버튼 간격 */
    width: 100%;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    padding: 0 10px;
}

/* 출처 버튼 컨테이너 */
.source-button-container {
    display: flex;
    flex: 0 0 auto; /* 크기 고정 */
}

/* 피드백 버튼들 컨테이너 */
.feedback-buttons {
    display: flex;
    gap: 12px; /* 버튼 간격 */
    justify-content: flex-end; /* 오른쪽 정렬 */
    flex: 0 0 auto; /* 크기 고정 */
}

/* 버튼 공통 스타일 */
.icon-button {
    box-sizing: border-box; /* 패딩과 보더 포함 */
    width: 85px; /* 고정 너비 */
    min-width: 85px;
    height: 36px; /* 고정 높이 */
    padding: 6px 12px; /* 좌우 패딩 증가로 텍스트 정렬 공간 확보 */
    font-size: 12.5px; /* 일관된 폰트 크기 */
    font-weight: 600; /* 글자 굵기 */
    border-radius: 12px; /* 둥근 모서리 */
    border: 1px solid rgba(0, 0, 0, 0.1); /* 테두리 */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* 그림자 */
    background-color: #ffffff; /* 기본 배경색 */
    color: #333333; /* 기본 텍스트 색상 */
    cursor: pointer; /* 마우스 커서 포인터 */
    transition: all 0.3s ease; /* 트랜지션 효과 */
    display: flex; /* Flexbox 사용 */
    justify-content: center; /* 수평 중앙 정렬 */
    align-items: center; /* 수직 중앙 정렬 */
    white-space: nowrap; /* 텍스트 줄바꿈 방지 */
    overflow: hidden; /* 넘치는 내용 숨기기 */
    text-overflow: ellipsis; /* 넘치는 텍스트 생략 부호 */
    text-align: center; /* 텍스트 중앙 정렬 */
    line-height: 1; /* 라인 높이 */
    margin: 0; /* 마진 제거 */
    -webkit-tap-highlight-color: transparent; /* 모바일 탭 하이라이트 제거 */
}

/* 출처 버튼 스타일 */
.source-button-container .icon-button {
    background-color: #ff8534; /* 출처 버튼 배경색 */
    color: white; /* 출처 버튼 텍스트 색상 */
    font-weight: 600; /* 출처 버튼 글자 굵기 */
}

.source-button-container .icon-button:hover {
    background-color: #ff7420; /* 호버 시 배경색 변경 */
    transform: translateY(-1px); /* 약간 위로 이동 */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* 호버 시 그림자 강화 */
}

/* 좋아요 버튼 스타일 */
.icon-button.like {
    background-color: #7171e5; /* 좋아요 버튼 배경색 */
    color: white; /* 좋아요 버튼 텍스트 색상 */
}

.icon-button.like:hover {
    background-color: #6161d5; /* 호버 시 배경색 변경 */
    transform: translateY(-1px); /* 약간 위로 이동 */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* 호버 시 그림자 강화 */
}

/* 싫어요 버튼 스타일 */
.icon-button.dislike {
    background-color: #66b2b2; /* 싫어요 버튼 배경색 */
    color: white; /* 싫어요 버튼 텍스트 색상 */
}

.icon-button.dislike:hover {
    background-color: #559999; /* 호버 시 배경색 변경 */
    transform: translateY(-1px); /* 약간 위로 이동 */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* 호버 시 그림자 강화 */
}

/* 버튼 내부 링크 스타일 */
.icon-button a {
    color: inherit; /* 부모 색상 상속 */
    text-decoration: none; /* 링크 밑줄 제거 */
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* 모바일 대응 */
@media screen and (max-width: 768px) {
    .feedback-container {
        gap: 8px; /* 간격 축소 */
        padding: 0 8px; /* 패딩 축소 */
        max-width: 500px; /* 최대 너비 조정 */
    }

    .icon-button {
        font-size: 12px; /* 폰트 크기 축소 */
        padding: 6px 8px; /* 패딩 유지 */
        width: 75px; /* 버튼 너비 축소 */
        min-width: 75px;
    }

    /* 출처 버튼의 패딩 조정 */
    .source-button-container .icon-button {
        padding: 6px 8px;
    }
}

/* 매우 작은 화면 대응 */
@media screen and (max-width: 480px) {
    .feedback-container {
        gap: 6px; /* 간격 더 축소 */
        padding: 0 6px; /* 패딩 더 축소 */
        max-width: 400px; /* 최대 너비 조정 */
    }

    .icon-button {
        font-size: 11.5px; /* 폰트 크기 축소 */
        padding: 4px 6px; /* 패딩 감소 */
        width: 65px; /* 버튼 너비 추가 축소 */
        min-width: 65px;
    }

    /* 출처 버튼의 패딩 조정 */
    .source-button-container .icon-button {
        padding: 4px 6px;
    }
}

/* 채팅 입력창 스타일 */
.chat-row textarea {
    border: 2px solid #f9d9c0 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    transition: all 0.3s ease;
}

.chat-row textarea:focus {
    border-color: #ffa976 !important;
    box-shadow: 0 0 0 2px rgba(187, 175, 189, 0.1) !important;
}

/* 채팅 제출 버튼 스타일 */
#chat-submit {
    background-color: #d8d8d6 !important;
    border: none !important;
    border-radius: 12px !important;
    color: #000 !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    letter-spacing: -0.3px !important;
    padding: 8px 16px; /* 일관된 패딩 */
}

#chat-submit:hover {
    background-color: #c8c8c6 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important; /* 호버 시 그림자 강화 */
}

/* 쿠키 동의 배너 스타일 */
#cookie-consent-banner {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 1rem;
    text-align: center;
    z-index: 99999;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, "Helvetica Neue", "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

#accept-cookies {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    background: #ffa976;
    color: white;
    cursor: pointer;
    transition: all 0.3s ease;
    font-family: inherit;
    font-weight: 600;
}

#accept-cookies:hover {
    background: #ff8534;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}
    """

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
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=8088, share=True,
        show_error=True,  # 에러 상세 표시
        show_api=False,   # API 엔드포인트 비활성화
        favicon_path=None # 기본 파비콘 사용)
    )
    