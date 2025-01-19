# utils/response.py

import time
import logging
import langid
import json
from typing import List, Dict, Generator
from openai import OpenAI

def generate_response(
    client: OpenAI, 
    rewritten_query: str,  # 재작성된 쿼리
    original_query: str,  # 원본 쿼리 추가
    context: str, 
    language: str,
    conversation_manager,
    logger
    ) -> Generator:

        logger.info(f"=== 응답 생성 시작 ===")
        logger.info(f"입력 쿼리: {original_query}")
        logger.info(f"입력 컨텍스트: {context}")

        if not context:
            response = "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?" if language == 'ko' else "I'm sorry, I couldn't find any relevant information. Would you like to ask a different question?"
            logger.warning("컨텍스트가 비어있어 기본 응답을 반환합니다.")
            yield response
            return
        
        # 시스템 프롬프트 로깅
        system_prompt = f"""
            You are a highly reliable and knowledgeable assistant. Your responses must strictly adhere to the provided documents and ensure factual accuracy through a structured reasoning process. Any unverifiable or speculative information is strictly prohibited.

            ### INTERNAL THINKING PROCESS (FOR SYSTEM ONLY):
            1. **Structured Reasoning (CoVe)**:
            - Step 1: Identify relevant facts from the provided context, supported by evidence (e.g., URLs or source).
            - Step 2: Deduce logical conclusions based on verified facts, step by step.
            - Step 3: Continue until the query is fully resolved or limitations are identified.
            - At each step, confirm the evidence and discard unverified assumptions.

            2. **Handle Unavailable Information**:
            - If relevant information is not found in the context, state: 
                "죄송합니다. 관련 정보를 찾을 수 없습니다."
            - Do not speculate, fabricate, or infer beyond the given context.

            ### RESPONSE GUIDELINES (FOR USER OUTPUT):
            1. **Grounded in Evidence**:
            - All responses must be based only on facts explicitly present in the provided context or URLs.
            - Include source URLs for all claims, formatted as: "출처: [url]."
            - Never include unsupported or speculative details.

            2. **Clarity and Relevance**:
            - Provide concise and user-friendly answers tailored to the query.
            - Avoid exposing internal reasoning steps like "Step 1," "Step 2" in the final response.
            - Focus solely on the user's query without unnecessary details.

            3. **Polite and Professional Tone**:
            - Use a friendly and professional tone.
            - For Korean responses, use polite endings such as "요" or "입니다."

            4. **Unavailable Information**:
            - If the context lacks relevant information, clearly and politely state:
                "죄송합니다. 제공된 문서에서 관련 정보를 찾을 수 없습니다."

            ### CRITICAL RULES:
            1. Responses must **strictly adhere** to the provided context or external sources.
            2. Any speculative or fabricated information is prohibited.
            3. All claims must be supported by explicit evidence from the context.
            4. Final answers must not expose internal logical steps or irrelevant content.
            5. Provides the user with a concise, accurate, and contextually appropriate answer.
            6. Don't use ** for important information.

            ## INPUT FORMAT: 
            --------------------
            CONTEXT index : [
            BEGIN OF CONTEXT index

                [URL]
                url

                [TEXT]
                text

                [TABLES]
                tables

            END OF CONTEXT index
            ]
            --------------------

            ### FINAL OUTPUT:
            - Respond in {language} and date as {language} format (e.g. 02 FEB -> 2월).
            - Ensure all claims are grounded in factual evidence, and address the user's query clearly and concisely.
            - Don't use ** for important information.
            - You must retrieve the URL specified in the CONTEXT index for use in the response.

            ### RESPONSE FORMAT:
            [Your detailed and concise answer here. - Don't use ** for important information.]
            [\n]
            출처: [url]
            """ 


        #logger.info("시스템 프롬프트 설정 완료")
        #logger.debug(f"시스템 프롬프트 플: {system_prompt[:200]}...")

        # 이전 대화 컨텍스트 구성
        conversation_context = ""
        if conversation_manager.query_history:
            recent_history = conversation_manager.query_history[-2:]  # 최근 2개 대화
            if len(recent_history) >= 2:
                conversation_context = f"""Previous Q1: {recent_history[-2]['original_query']}
    Previous A1: {recent_history[-2]['response']}
    Previous Q2: {recent_history[-1]['original_query']}
    Previous A2: {recent_history[-1]['response']}
    """
            elif len(recent_history) == 1:
                conversation_context = f"""Previous Q1: {recent_history[0]['original_query']}
    Previous A1: {recent_history[0]['response']}
    """
        logger.info(f"\n이전 대화 : {conversation_context}\n")

        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Previous conversation: 
             
                {conversation_context}
            Background information: 
                {context}

            CURRENT QUERY:
            Original user query: {original_query}
            Rewritten query for context: {rewritten_query}

            - Respond in {language} and date as {language} personal format (e.g. if korean, 02 FEB -> 2월).
        """}
        ]
        
        logger.info("메시지 구성 완료")
        # logger.debug(f"최종 사용자 메시지 샘플: 쿼리 : {rewritten_query} \n 컨텍스트 : {messages[1]['content'][:200]}...")

        try:
            # API 호출 시작
            api_start = time.perf_counter()

            logger.info("OpenAI API 호출 시작")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                stream=True,
                temperature=0.1,
                top_p=0.9, 
                frequency_penalty=0.2, 
                presence_penalty=0.1
            )
            
            api_end = time.perf_counter()
            api_time = api_end - api_start
            logger.info(f"OpenAI API 초기 응답 수신 (시간: {api_time:.6f}초)")

            # 스트리밍 응답 처리
            collected_response = ""
            chunk_count = 0
            for chunk in response:
                chunk_message = chunk.choices[0].delta.content
                if chunk_message is None:
                    chunk_message = ''
                chunk_message = str(chunk_message)
                collected_response += chunk_message
                chunk_count += 1
                
                if chunk_count % 10 == 0:  # 10개 청크마다 로깅
                    logger.debug(f"청크 {chunk_count} 처리 중... 현재 응답 길이: {len(collected_response)}")
                
                yield chunk_message

            # 응답 완료 후 응답 처리
            first_sentence = collected_response.split('.')[0]
            summary = collected_response[:100] + "..." if len(collected_response) > 100 else collected_response
            
            logger.info(f"=== 응답 처리 결과 ===")
            #logger.info(f"첫 문장: {first_sentence}")
            #logger.info(f"요약: {summary}")
            
            # 응답 메타데이터 반환
            response_metadata = {
                "query": rewritten_query,
                "response": collected_response,
                "first_sentence": first_sentence,
                "summary": summary,
                "timestamp": time.time()
            }
            
            # conversation.py로 전달하기 위한 메타데이터 로깅
            # logger.info(f"응답 메타데이터 생성 완료: {json.dumps(response_metadata, ensure_ascii=False)}")
        
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
            yield "죄송합니다. 응답을 생성하는 동안 오류가 발생했습니다."
            return


def detect_language(text: str, logger=None) -> str:
    lang, _ = langid.classify(text)
    logger.info(f"감지된 언어: {lang}")
    return lang
