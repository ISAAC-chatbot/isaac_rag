# utils/conversation.py

import time
import logging
import json
from typing import List, Literal, TypedDict, Optional
from typing_extensions import Annotated
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    BaseMessage as Message
)
from openai import OpenAI
from pydantic import BaseModel
from .search import hybrid_search
from .embeddings import generate_query_vector


class ConversationMessage(BaseModel):
    """대화 메시지 모델"""
    role: str
    content: str
    timestamp: float

class ConversationState(TypedDict):
    """대화 상태를 나타내는 타입"""
    messages: Annotated[List[Message], "messages"]
    summary: str
    query_history: List[dict]
    rewritten_query: Optional[str]
    current_query: Optional[str]
    documents : List[dict]
    context: str
    language: Optional[str]
    needs_response: bool  # 응답 생성 필요 여부
    subquery: Optional[str]
    topics: Optional[List[str]]

class ConversationManager:
    def __init__(self, client: OpenAI, max_queries: int = 5, search_config: dict = None, logger=None):
        self.client = client
        self.max_queries = max_queries
        self.search_config = search_config or {}
        self.query_history = []
        self.graph = self._create_conversation_graph()
        self.logger = logger or logging.getLogger(__name__)

    def _create_conversation_graph(self) -> StateGraph:
        workflow = StateGraph(ConversationState)
        
        # 노드 정의
        workflow.add_node("get_query", self._get_query)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("search_documents", self._search_documents)
        workflow.add_node("create_context", self._create_context)
        workflow.add_node("generate_response_virtual", self._generate_response_virtual)
        workflow.add_node("manage_history", self._manage_history)
        
        
        # 엣지 설정
        workflow.add_edge(START, "get_query")
        workflow.add_edge("get_query", "rewrite_query")
        workflow.add_edge("rewrite_query", "search_documents")
        workflow.add_edge("search_documents", "create_context")
        workflow.add_edge("create_context", "generate_response_virtual")
        workflow.add_conditional_edges(
            "generate_response_virtual",
            self._should_remove,
            {
                True: "manage_history",
                False: END
            }
        )
        workflow.add_edge("manage_history", END)
        
        return workflow.compile()

    def _get_query(self, state: ConversationState) -> ConversationState:
        """쿼리 상태 확인 노드"""
        if not state.get("messages"):
            self.logger.warning("메시지가 없는 상태로 진행됩니다.")
            return state
            
        self.logger.info("쿼리 상태 확인 완료")
        return state
    
    def extract_block(self, lines, start_marker, stop_markers):
            """
            lines: 응답 전체를 줄 단위로 나눈 리스트
            start_marker: 블록 시작 문자열 (예: "REWRITTEN:")
            stop_markers: 블록 종료를 판단할 다른 시작 문자열 리스트 (예: ["EXPECTED_DOC:", "TOPICS:"])

            start_marker가 발견된 이후, 다음 중 하나라도 발견될 때까지의 모든 줄을 합쳐서 반환합니다.
            """
            content_lines = []
            capturing = False
            for line in lines:
                if line.startswith(start_marker):
                    # 블록 시작 구간
                    capturing = True
                    # 첫 줄에서 start_marker 부분은 제거
                    content_lines.append(line.split(':', 1)[1].strip())
                    continue
                if capturing:
                    # 다른 마커를 만나면 블록 종료
                    if any(line.startswith(m) for m in stop_markers):
                        break
                    # 블록 내부 내용 계속 수집
                    content_lines.append(line)

            # 수집된 라인들을 하나의 문자열로 합침
            return "\n".join(content_lines).strip()

    def _rewrite_query(self, state: ConversationState) -> ConversationState:
        """쿼리 재작성 및 주제 추출"""
        self.logger.info("=== Query Rewrite Process Started ===")

        if not state.get("current_query"):
            self.logger.warning("현재 쿼리가 없습니다.")
            return state

        current_query = state.get("current_query")
        self.logger.info(f"Current Query: {current_query}")
        self.logger.info(f"Query History Count: {len(self.query_history)}")


        json_file_path = "/home/ubuntu/multiturn/utils/query_embeddings.json"

        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                existing_embeddings = json.load(f)
                if current_query in existing_embeddings:
                    self.logger.info(f"Query '{current_query}' already exists in the embeddings file.")
                    return {
                **state,
                "rewritten_query": current_query,
                "subquery": current_query,
                "topics": [],
                "current_query" : current_query
            }

        except FileNotFoundError:
            self.logger.warning(f"JSON file '{json_file_path}' not found. Proceeding with embedding generation.")
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON file '{json_file_path}'. Proceeding with embedding generation.")

        # 이전 쿼리들을 순서대로 컨텍스트 구성
        recent_history = []
        for entry in self.query_history[-4:]:  # 최근 4개 대화
            history_str = (
                f"Q: {entry['original_query']}\n"
                f"A: {entry['response']}"
            )
            recent_history.append(history_str)
            self.logger.info(history_str)

        system_prompt = """
        You are a query rewriter, topic analyzer, and document hypothesis generator specialized for Yonsei University's chatbot. Your responsibilities are:

        ### TASKS:
        1. **Query Rewriting**  
            - Rewrite the user query for clarity, relevance, and expanded vocabulary.  
            - Incorporate synonymous terms (e.g., "학점," "성적," "평점").  
            - Use the previous interactions to understand the full context and ensure your rewritten query aligns with the user's intent, even if implied or indirect.  
        2. **Document Hypothesis (EXPECTED_DOC)**  
            - Infer and generate a JSON structure representing the expected document content based on the query context.  
        3. **Topic Analysis**  
            - Select the most relevant topics from predefined categories.  

        ### CONTEXT ANALYSIS:
        1. **Analyze Previous Interactions**  
            - Use up to the last 3 queries and responses for contextual understanding.  
            - Extract recurring terms, focus areas, and user intent from the conversation history to clarify ambiguous queries or implicit references.  
            - Maintain continuity between queries and avoid abrupt topic shifts unless explicitly indicated.  
        2. **Query Classification**  
            - Determine if the current query is:  
                a. A follow-up  
                b. A clarification request  
                c. A topic shift  
                d. A new topic.  
        3. **Temporal Context Rules**  
            - Default to "2025-1" for schedule-related queries unless explicitly stated otherwise (like 기한, 마감, 입사 등).
            - Use explicitly mentioned semesters (e.g., "2024-2").
            - Unless explicitly stated otherwise, such as mentioning '대학원', treat queries as referring to '학부' or '학부대학' by default.
            - If unrelated to schedules, omit temporal context unless explicitly required.  

        ### TOPIC ANALYSIS:
        1. **Predefined Topics**  
            predefined_categories = {
                "computing.yonsei.ac.kr": "인공지능융합대학",
                "cs.yonsei.ac.kr":        "인공지능융합대학",
                "dorm.yonsei.ac.kr":      "연세대생활관",
                "oia.yonsei.ac.kr":       "국제처",
                "yicdorm.yonsei.ac.kr":   "송도학사",
                "yicrc.yonsei.ac.kr":     "RC 교육원",
                "yonsei.ac.kr":           "연세대전반",
                "ysb.yonsei.ac.kr":       "경영대학"
            }
        2. **Topic Selection Rules**  
            - Identify topics based on the query and its context.  
            - Prioritize topics mentioned in the previous interactions if they align with the current query.  
            - Limit output to **2 topics maximum** if ambiguity exists.  
            - Ensure topics match the predefined categories.  

        ### RESPONSE FORMAT:
        REWRITTEN: [Rewritten query with enhanced clarity and context]
        EXPECTED_DOC:  
            ```json
            {
                "title":              "[Inferred document title]",
                "summary":            "[1-sentence summary of the expected document]",
                "keywords for DOC":   "[5 related keywords]",
                "expected_queries":   [
                                        "[Example query 1]",
                                        "[Example query 2]",
                                        "[Example query 3]"
                                    ]
            }
            ```
        TOPICS: [List of relevant topics, e.g., ["경영대학", "연세대전반"]]

        ### EXAMPLES:

        1. **Follow-Up Query**
        Previous: "컴퓨터과학과 복수전공 졸업요건?"
        Current: "경영학과는?"
        → REWRITTEN: "경영학과 학부 복수전공 졸업요건"
        → EXPECTED_DOC: {...}
        → TOPICS: ["경영대학"]

        2. **Follow-Up Query 2**
        Previous: "현재 총장님 성함 알려줘."
        Current: "그럼 경영학과는?"
        → REWRITTEN: "연세대학교 경영학과 총장은 누구인가요?"
        → EXPECTED_DOC: {...}
        → TOPICS: ["경영대학"]

        3. **Implicit Temporal Context**
        Previous: "수강신청 언제야?"
        Current: "정정은?"
        → REWRITTEN: "2025-1 학부 수강신청 정정기간"
        → EXPECTED_DOC: {...}
        → TOPICS: ["연세대전반"]

        4. **Explicit Temporal Reference**
        Previous: "2024-2 교환학생 신청 기간 알려줘."
        Current: "2025-1도 같은 기간이야?"
        → REWRITTEN: "2025-1 학부 교환학생 신청 기간"
        → EXPECTED_DOC: {...}
        → TOPICS: ["국제처", "연세대전반"]

        5. **SCHEDULE-RELATED QUERY WITH EXPLICIT TEMPORAL SHIFT**  
        Previous: "이번학기 등록금 언제까지야?"  
        Current: "다음 학기는?"  
        → REWRITE: "2025-1 학부 등록금 납부 기간"   
        → EXPECTED_DOC: {...}
        → TOPICS: ["연세대전반"] 

        7. **NON-SCHEDULE QUERY FOLLOW-UP**  
        Previous: "기숙사 방 크기가 어때?"  
        Current: "공간이 더 넓은 방은 없어?"  
        → REWRITE: "기숙사 공간이 넓은 방 옵션"  
        → EXPECTED_DOC: {...}
        → TOPICS: ["송도학사", "연세대생활관"]
        

        ### ADDITIONAL INSTRUCTIONS:
        - When in doubt, prioritize clarity and maintaining the user's intended flow of thought.
        - If a query is ambiguous, attempt to infer the user's intent based on prior interactions before introducing new topics.

        Provide outputs in the specified format with precise adherence to rules.
        """



        user_prompt = f"""PREVIOUS INTERACTIONS (Most recent first):
                    {chr(10).join(recent_history)}

                    CURRENT QUERY: {current_query}

                    Analyze the conversation and respond with the specified format."""
                                        

        try:
            self.logger.info("Calling GPT-4o for query rewriting")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.05,
                top_p=0.9, 
                frequency_penalty=0.2, 
                presence_penalty=0.05
            )

            result = response.choices[0].message.content
            self.logger.info(f"GPT-4o Response: {result}")

            lines = result.split('\n')

            # 우선순위: REWRITTEN, EXPECTED_DOC, TOPICS
            rewritten_block = self.extract_block(lines, "REWRITTEN:", ["EXPECTED_DOC:", "TOPICS:"])
            expected_doc_block = self.extract_block(lines, "EXPECTED_DOC:", ["REWRITTEN:", "TOPICS:"])
            topics_block = self.extract_block(lines, "TOPICS:", ["REWRITTEN:", "EXPECTED_DOC:"])

            rewritten = rewritten_block if rewritten_block else None
            subquery = expected_doc_block if expected_doc_block else None
            topics = None

            # 주제 문자열을 쉼표로 분리
            if topics_block:
                # "TOPICS:" 다음에 대괄호가 있을 수도 있으므로 제거
                topics_clean = topics_block.strip().strip('[]')
                # 쉼표 단위로 나누고, 양 끝 공백/따옴표 제거
                topics = [
                    t.strip().strip('"').strip() for t in topics_clean.split(',')
                    if t.strip()
                ]

            self.logger.info(f"Query Rewrite Process Completed \n rewritten : {rewritten} \n subquery : {subquery} \n topics : {topics}")

            if not rewritten:
                self.logger.warning("Failed to parse rewritten query, using original")
                rewritten = current_query

            return {
                **state,
                "rewritten_query": rewritten,
                "subquery": subquery,
                "topics": topics,
                "current_query" : current_query
            }

        except Exception as e:
            self.logger.error(f"Query rewriting error: {str(e)}")
            return {
                **state,
                "rewritten_query": current_query,
                "subquery": current_query,
                "topics": [],
                "current_query" : current_query
            }

    def update_search_config(self, new_search_config: dict):
        """검색 설정을 업데이트합니다."""
        self.logger.info(f"새 검색 설정으로 업데이트: {new_search_config}")
        self.search_config.update(new_search_config)

    def _search_documents(self, state: ConversationState) -> ConversationState:
        """문서 검색 노드"""
        self.logger.info("=== 문서 검색 시작 ===")
        rewritten_query = state.get("rewritten_query")
        subquery = state.get("subquery")
        current_query = state.get("current_query")
        search_method = self.search_config.get("search_method", "ISAAC Harmony - 균형 잡힌 검색")
        lang = state.get("language", "ko")

        if search_method == 'ISAAC Pro - 자세한 검색':
            query_embedding = generate_query_vector(self.client, subquery, current_query ,self.logger)
        else :
            query_embedding = generate_query_vector(self.client, rewritten_query, current_query ,self.logger)
        
        # hybrid-pipeline-vector-focused 1:9
        # hybrid-pipeline-text-focused 9:1
        # hybrid-pipeline-balanced 5:5

        # 검색 방법 매핑
        search_method_mapping = {
            'ISAAC Basic - 균형 잡힌 검색': 'hybrid-pipeline-balanced',
            # 'ISAAC Insight - 의미 기반 검색': 'hybrid-pipeline-text-focused',
            # 'ISAAC Classic - 키워드 우선 검색': 'hybrid-pipeline-vector-focused',
            'ISAAC Pro - 자세한 검색': 'hybrid-pipeline-balanced'
        }

        # 매핑된 검색 방법 가져오기
        actual_search_method = search_method_mapping.get(search_method)
        if not actual_search_method:
            self.logger.error(f"알 수 없는 검색 방법: {search_method}")
            error_message = "죄송합니다. 내부 오류가 발생했습니다." if lang == 'ko' else "I'm sorry, an internal error occurred."
        
        try: 
            if actual_search_method in ['hybrid-pipeline-balanced', 'hybrid-pipeline-text-focused', 'hybrid-pipeline-vector-focused']:
                documents = hybrid_search(
                    user_query_text = rewritten_query,
                    user_query_vector = query_embedding,
                    top_k = self.search_config.get("top_k", 5),
                    pipeline_name = actual_search_method,
                    logger = self.logger
                )
                # user_query_text, user_query_vector, k=5, pipeline_name="hybrid-pipeline-balanced"
                # self.logger.info(f"검색 결과를 사용하여 document : {documents[0]}를 검색했습니다.")
            else:
                self.logger.error(f"알 수 없는 검색 방법: {search_method}")
                error_message = "죄송합니다. 내부 오류가 발생했다." if lang == 'ko' else "I'm sorry, an internal error occurred."
                return {
                    **state,
                    "documents": [],
                    "response": error_message
                }
            return {
                **state,
                "documents": documents
            }
        except Exception as e:
            self.logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            error_message = "죄송합니다. 문서 검색 중 오류가 발생했습니다." if lang == 'ko' else "I'm sorry, an error occurred during document search."
            return {
                **state,
                "documents": [],
                "response": error_message
            }


    def _create_context(self, state: ConversationState) -> ConversationState:

        # self.logger.info(f"State 내용: {state}")
        document = state.get("documents", [])
        max_length = 50000
        self.logger.info("AI를 위한 컨텍스트 텍스트 생성 시작.")
        context_text = ""
        index = 0
        for doc in document:
            text = doc.get('merged_text', 'N/A')
            tables = doc.get('tables', 'N/A')
            url = doc.get('url', 'Unknown URL')
            doc_text = f"""
                        --------------------
                        CONTEXT {index} : [
                        BEGIN OF CONTEXT {index}

                            [URL]
                            {url}

                            [TEXT]
                            {text}

                            [TABLES]
                            {tables}

                        END OF CONTEXT {index}
                        ]
                        --------------------
                        """

            index += 1
            if len(context_text) + len(doc_text) > max_length:
                self.logger.info(f"문서 길이 : { len(context_text) + len(doc_text)}\n")
                remaining_length = max_length - len(context_text)
                if remaining_length > 0:
                    truncated_doc_text = doc_text[:remaining_length]
                    context_text += truncated_doc_text
                    # logging.info(f"문서 {url} 일부를 잘라서 추가했습니다.\n 문서의 첫 500자입니다 ： {context_text[:500]}\n")
                break
            context_text += doc_text
            # logging.info(f"추가된 문서: {doc_text}...\n")
        self.logger.info("컨텍스트 텍스트 생성 완료.")
        # logging.info(f"생성된 컨텍스트 텍스트 길이: {len(context_text)}")
        self.logger.info(f"생성된 컨텍스트 텍스트 내용: {context_text}...\n")
        return {
                **state,
                "context": context_text
            }


    def _generate_response_virtual(self, state: ConversationState) -> ConversationState:
            """가상 응답 생성 노드 - 실제 응답 생성은 main.py에서 처리"""
            self.logger.info("=== 가상 응답 생성 노드 도달 ===")
            # 이 노드에서는 실제 응답 생성을 하지 않고, main.py에서 처리하도록 플래그 설정
            state["needs_response"] = True
            return state


    def _should_remove(self, state: ConversationState) -> bool:
        """대화 히스토리가 최대 크기를 초과하는지 확인"""
        query_history = state.get("query_history", [])
        return len(query_history) >= self.max_queries


    def _manage_history(self, state: ConversationState) -> ConversationState:
        """대화 히스토리 관리 - 가장 오래된 대화 제거"""
        self.logger.info("=== 대화 히스토리 관리 시작 ===")
        query_history = state.get("query_history", [])
        
        if len(query_history) >= self.max_queries:
            # 가장 오래된 대화를 제거하고 나머지를 한 칸씩 앞으로 이동
            query_history = query_history[1:]  # 첫 번째 항목 제거
            self.logger.info(f"가장 오래된 대화 제거 완료. 현재 대화 수: {len(query_history)}")
        
        return {
            **state,
            "query_history": query_history
        }


    def process_message(self, message: str, language: str = "ko") -> ConversationState:
        """사용자 메시지를 처리하고 최종 상태를 반환"""
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "summary": "",
            "query_history": self.query_history.copy(),
            "rewritten_query": None,
            "current_query": message,
            "context": [],
            "language": language,
            "needs_response": False
        }

        # run -> invoke로 메서드명 변경
        final_state = self.graph.invoke(initial_state)

        # needs_response 플래그 설정 시, 응답 생성을 main.py에서 처리하도록 함
        if final_state.get("needs_response"):
            final_state["needs_response"] = True

        return final_state


    def update_query_history(self, original_query: str, rewritten_query: str, response: str):
        """대화 기록 업데이트"""
        # 중복 기록 방지를 위한 검사
        if self.query_history and self.query_history[-1].get("rewritten_query") == rewritten_query:
            self.logger.info("동일한 쿼리가 이미 기록되어 있어 건너뜁니다.")
            return
        
        # 최대 크기를 초과하는 경우 가장 오래된 대화 제거
        if len(self.query_history) >= self.max_queries:
            self.query_history = self.query_history[1:]  # 첫 번째 항목 제거
            self.logger.info("가장 오래된 대화 제거됨")
        
        conversation_id = len(self.query_history) + 1
        conversation_entry = {
            "id": conversation_id,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "response": response,
            "timestamp": time.time()
        }
        self.query_history.append(conversation_entry)
        self.logger.info(f"대화 #{conversation_id} 기록됨:\n{json.dumps(conversation_entry, indent=2, ensure_ascii=False)}")
