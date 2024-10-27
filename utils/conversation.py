# utils/conversation.py

import time
import logging
import numpy as np
from .embeddings import embed_text  # embedding_cache 제거
import re

# 후속 질문 패턴 정의
FOLLOW_UP_PATTERNS = [
    "다시", "아니", "그럼", "추가로", "또", "그리고",
    "근데", "그니까", "그래서", "아 맞다", "아, 그럼", "아 그리고", "그렇다면", "더",
    "더 구체적으로", "구체적으로",
    # 필요한 만큼 추가...
]



#제외할 키워드
EXCLUDE_KEYWORDS = [
    "알려줘", "안내", "정보", "질문", "도와줘", "알려주세요", "언제야", "어디야",
    "알려주라", "가르쳐줘", "가르쳐주세요", "어떻게", "뭐야", "누구야", "어디서",
    "언제부터", "언제까지", "말해줘", "말해주세요", "설명해줘", "설명해주세요",
    "좀", "제발", "부탁해", "부탁합니다", "싶어", "싶어요", "필요해", "필요합니다", "정리해줘",
    "정리", "알려주세요"
    # 필요한 만큼 추가...
]


def preprocess_message(message: str) -> str:
    """
    메시지에서 제외할 키워드를 제거합니다.
    """
    pattern = '|'.join(map(re.escape, EXCLUDE_KEYWORDS))
    cleaned_message = re.sub(pattern, '', message)
    return cleaned_message.strip()

def is_follow_up_question(question: str) -> bool:
    """후속 질문 여부를 확인합니다."""
    return any(question.strip().startswith(pattern) for pattern in FOLLOW_UP_PATTERNS)

class ConversationManager:
    def __init__(self, max_length=5, base_similarity_threshold=0.83):
        self.history = []
        self.history_embeddings = []
        self.max_length = max_length
        self.base_similarity_threshold = base_similarity_threshold
        self.last_response_time = None

    def add_message(self, client, message):
        current_time = time.time()

        # 메시지 전처리
        preprocessed_message = preprocess_message(message)
        logging.info(f"전처리된 메시지: {preprocessed_message}")

        # 후속 질문인지 확인하여 유사도 계산 생략
        if is_follow_up_question(preprocessed_message):
            logging.info("후속 질문 감지: 기존 대화 맥락을 그대로 유지합니다.")
            self._add_to_history(message, client)
            return

        dynamic_threshold = self.base_similarity_threshold

        # 시간 기반 임계값 조정
        if self.last_response_time:
            time_diff = current_time - self.last_response_time
            logging.info(f"메시지 간 시간 차이 (사용자 입력 - 이전 응답 완료): {time_diff:.2f}초")

            if time_diff <= 3.3:
                dynamic_threshold -= 0.1
                dynamic_threshold = max(dynamic_threshold, 0.73)  # 최소값 제한
            else:
                dynamic_threshold += 0.005
                dynamic_threshold = min(dynamic_threshold, 0.84)  # 최대값 제한

        # 대화 맥락의 전처리된 결합
        preprocessed_history = [preprocess_message(msg) for msg in self.history]
        combined_context = ' '.join(preprocessed_history)

        # 임베딩 생성
        if combined_context:
            combined_embedding = embed_text(client, [combined_context])[0]
            logging.info("전처리된 결합된 맥락 임베딩을 생성하거나 캐시에서 불러왔습니다.")
        else:
            combined_embedding = None  # 빈 맥락일 경우 None으로 설정

        if preprocessed_message:
            current_embedding = embed_text(client, [preprocessed_message])[0]
            logging.info("전처리된 현재 메시지 임베딩을 생성하거나 캐시에서 불러왔습니다.")
        else:
            current_embedding = None  # 전처리 후 메시지가 비어 있을 경우

        # 유사도 계산
        if combined_embedding is not None and current_embedding is not None:
            similarity = np.dot(combined_embedding, current_embedding) / (
                np.linalg.norm(combined_embedding) * np.linalg.norm(current_embedding)
            )
            logging.info(f"전처리된 맥락과 현재 메시지 간 유사도: {similarity:.2f}, 동적 임계값: {dynamic_threshold:.2f}")

            if similarity < dynamic_threshold:
                # 주제 전환 감지
                self.history = []
                self.history_embeddings = []
                logging.info("주제 전환 감지: 대화 맥락을 초기화합니다.")

        # 메시지 추가
        self._add_to_history(message, client)

    def _add_to_history(self, message, client):
        self.history.append(message)
        message_embedding = embed_text(client, [preprocess_message(message)])[0]
        self.history_embeddings.append(message_embedding)

        if len(self.history) > self.max_length:
            self.history.pop(0)
            self.history_embeddings.pop(0)

    def update_response_time(self):
        self.last_response_time = time.time()

    def get_context(self):
        return ' '.join(self.history)