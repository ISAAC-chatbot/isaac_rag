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