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