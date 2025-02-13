from fastapi import FastAPI, Request, BackgroundTasks, Header, HTTPException, Depends, APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from utils.logging_utils import (
    log_query,
    get_logger_for_user)
from utils.response import (generate_response, detect_language)
from utils.conversation import ConversationManager
from utils.data_loader import initialize_client

import logging
import re
import html
from typing import List, Generator
import json
import aiohttp
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# BACKEND_SERVER 변수 가져오기
BACKEND_SERVER = os.getenv("BACKEND_SERVER")  # 기본값 설정 가능

router = APIRouter()

async def update_history(token: str, question: str, answer: str, source: str):

    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{BACKEND_SERVER}/api/v1/chat/histories",
                json={"question": question, "answer": answer, "sourceURL": source},
                headers={"Authorization": f"Bearer {token}"}
            )
    except Exception as e:
        print(f"History 저장 중 오류 발생: {str(e)}")

def get_bearer_token(Authorization: str = Header(None)) -> str:
    
    if not Authorization:
        raise HTTPException(status_code=401, detail="Authorization 헤더가 없습니다.")

    # "Bearer {토큰}" 형식 확인
    token_match = re.match(r"^Bearer\s+(.+)", Authorization)
    
    if not token_match:
        raise HTTPException(status_code=401, detail="잘못된 Authorization 형식입니다. 'Bearer {토큰}'이어야 합니다.")

    return token_match.group(1)

class HistoryRequest(BaseModel):
    original_query: str
    response: str

class ChatRequest(BaseModel):
    histories : List[HistoryRequest]
    message: str
    search_method: str
    user_id : str

class ChatResponse(BaseModel):
    source: bool  # URL 또는 출처 정보가 포함되었는지 여부
    text: str    # 실제 텍스트 또는 URL
    last: bool    # 마지막 청크인지 여부

@router.post("/chat", response_model=ChatResponse, responses={200: {"content": {"application/json": {"example": {"source": False, "text": "url 또는 청크단위의 텍스트", "last": False}}}}})
def chat(
    request: ChatRequest, 
    fastapi_request: Request, 
    background_tasks: BackgroundTasks, 
    token: str = Depends(get_bearer_token)):

    session_managers = {}

    client = initialize_client()

    user_message = request.message
    search_method_selection = request.search_method
    user_id = request.user_id
    # Pydantic 객체 리스트를 딕셔너리 리스트로 변환
    histories = [history.model_dump() if isinstance(history, BaseModel) else history for history in request.histories]


    def response_generator() -> Generator[str,None, None]:
        logger = get_logger_for_user(user_id) or logging.getLogger()
        logger.info(f"[사용자 {user_id}] 사용자 쿼리: {user_message}")

        # Log the query
        log_query(user_id, user_message, search_method_selection)
        logger.info(f"선택된 검색 방법: {search_method_selection}")

        # 세션별 ConversationManager 가져오기 또는 생성
        if user_id not in session_managers:
            session_managers[user_id] = ConversationManager(
                client=client,
                max_queries=6,
                search_config={
                    "search_method": search_method_selection,
                    "top_k" : 5
                },
                logger = logger
            ) 
        else:
            # 기존 ConversationManager의 검색 설정 업데이트
            session_managers[user_id].update_search_config({
                "search_method": search_method_selection
            })
        
        conversation_manager = session_managers[user_id]
        
        # 언어 감지
        language = detect_language(user_message, logger)
        
        # 로그 기록
        logger.info(f"새로운 쿼리 수신: {user_message} (언어: {language})")

        # 히스토리 지정
        conversation_manager.query_history = histories
        
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
                data = ChatResponse(source=False, text=chunk, last=False)
                yield json.dumps(data.model_dump()) + "\n"
          

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
            
            data = ChatResponse(source=True, text=clean_url_escape, last=True)
            yield json.dumps(data.model_dump()) + "\n"
            # api에서는 이벤트 리스너 발생

        else:
            # 요약 등 다른 처리가 필요한 경우
            bot_response = final_state.get("response", "죄송합니다. 응답을 생성할 수 없습니다.")
            clean_url_escape= ""            
            data = ChatResponse(source=False, text=bot_response, last=True)
            yield json.dumps(data.model_dump()) + "\n"
        
        # 비동기 이벤트 리스너
        background_tasks.add_task(update_history, token, user_message, bot_message, clean_url_escape)
    return StreamingResponse(response_generator(), media_type="application/json")
