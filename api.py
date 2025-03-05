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
from typing import List, Generator, Optional, Union
import json
import requests
import os
from dotenv import load_dotenv
from enum import Enum
from datetime import datetime
import time

# .env 파일 로드
load_dotenv()

# BACKEND_SERVER 변수 가져오기
BACKEND_SERVER = os.getenv("BACKEND_SERVER")  # 기본값 설정 가능

router = APIRouter()

def update_history(token: str, chat_room_id: Optional[int], question: str, answer: str, source: str, elapsed_time: float):
    try:
        payload = {
            "question": question,
            "answer": answer,
            "sourceURL": source,
            "elapsedTime" : elapsed_time
        }

        # ✅ chat_room_id가 None이 아닐 때만 추가
        if chat_room_id is not None:
            payload["chatRoomId"] = chat_room_id

        response = requests.post(
            f"{BACKEND_SERVER}/api/v1/chat/messages",
            json=payload,
            headers={"Authorization": f"Bearer {token}"}
        )

        return response.json()  # 응답 반환

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

class ResponseType(str, Enum):
    MESSAGE = "MESSAGE"
    URL = "URL"
    CHAT_ROOM_INFO = "CHAT_ROOM_INFO"

class HistoryRequest(BaseModel):
    original_query: str
    response: str

class ChatRequest(BaseModel):
    histories : List[HistoryRequest]
    chat_room_id: Optional[int] = None
    message: str
    search_method: str
    user_id : str

class ChatResponse(BaseModel):
    type: ResponseType
    text: str

class ChatRoomResponse(BaseModel):
    type: ResponseType
    id : int
    title : str
    createdAt : datetime

@router.post(
    "/chat",
    response_model=Union[ChatResponse, ChatRoomResponse],  
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "chat_response_message": {
                            "summary": "채팅 메시지 응답",
                            "value": {
                                "type": "MESSAGE",
                                "text": "청크 단위의 텍스트"
                            }
                        },
                        "chat_response_url": {
                            "summary": "출처 사이트 응답",
                            "value": {
                                "type": "URL",
                                "text": "https://www.yonsei.ac.kr/sc/support/calendar.jsp?cYear=2025&amp;hakGi=1"
                            }
                        },
                        "chatroom_response": {
                            "summary": "채팅방 정보 응답",
                            "value": {
                                "type": "CHAT_ROOM_INFO",
                                "id": 1,
                                "title": "채팅방 제목",
                                "createdAt": "2025-02-14T12:00:00"
                            }
                        }
                    }
                }
            }
        }
    }
)
def chat(
    request: ChatRequest, 
    token: str = Depends(get_bearer_token)):

    start_time = time.time()

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

                if chunk == "출":
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

                if "출처:" in buffer or "출처 :" in buffer:
                    is_source_section = True
                    buffer = ""
                    continue

                if is_source_section:
                    url_message += chunk.strip()
                    continue

                if not first_flag and not second_flag and not is_source_section:
                    bot_message += chunk


                data = ChatResponse(type=ResponseType.MESSAGE, text=chunk)
                yield f"data: {data.json()}\n\n"
                # yield json.dumps(data.model_dump()) + "\n"

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
            
            data = ChatResponse(type=ResponseType.URL, text=clean_url, last=True)
            yield f"data: {data.json()}\n\n"
            # yield json.dumps(data.model_dump()) + "\n"
            
             # 최종 응답 직전 시간 측정
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 3)
            history_response = update_history(token, request.chat_room_id, user_message, bot_message, clean_url_escape, elapsed_time)
            
            # JSON 응답을 dict로 변환 후 'last': True 추가
            if isinstance(history_response, dict):  
                history_response["type"] = ResponseType.CHAT_ROOM_INFO
            else:
                history_response = {"type" : ResponseType.CHAT_ROOM_INFO, "response": history_response}  # 응답이 dict가 아닐 경우 기본 구조 생성

            history_response_data = json.dumps(history_response)
            yield f"data: {history_response_data}\n\n"
            # yield history_response_data + "\n"

        else:
            bot_response = final_state.get("response", "죄송합니다. 응답을 생성할 수 없습니다.")
            clean_url_escape= ""            
            data = ChatResponse(type=ResponseType.MESSAGE, text=bot_response, last=True)
            yield f"data: {data.json()}\n\n"
            # yield json.dumps(data.model_dump()) + "\n"
            
            history_response = update_history(token, request.chat_room_id, user_message, bot_response, clean_url_escape)
            # JSON 응답을 dict로 변환 후 'last': True 추가
            if isinstance(history_response, dict):
                history_response["type"] = ResponseType.MESSAGE  
            else:
                history_response = {"type" : ResponseType.CHAT_ROOM_INFO, "response": history_response}  # 응답이 dict가 아닐 경우 기본 구조 생성

            history_response_data = json.dumps(history_response)
            yield f"data: {history_response_data}\n\n"
            # yield history_response_data + "\n"

        # 비동기 이벤트 리스너
        # background_tasks.add_task(update_history, token, request.chat_room_id, user_message, bot_message, clean_url_escape)
    return StreamingResponse(response_generator(), media_type="text/event-stream")


