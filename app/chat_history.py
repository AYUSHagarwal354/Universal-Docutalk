"""This file is responsible for storing chat responses when session is up and deleting them when the session is closed

*** It is highly responsible for the conversational flow"""

import redis
import json
import os
from typing import Optional
import logging
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests
import time

# HISTORY_SIZE = int(os.getenv("HISTORY_SIZE", 3))

REDIS_HOST = os.getenv("REDIS_HOST","localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_DECODE_RES = os.getenv("REDIS_DECODE_RES", "true") == "true"
REDIS_USERNAME: Optional[str] = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD") or None

API_TOKEN_IBM = os.getenv("API_TOKEN_IBM","o8l7Q4LcFCCyZpOrJAdxjhjpCEJgZXfjrmo9VINP2cnA")
PROJECT_ID_IBM = os.getenv("PROJECT_ID_IBM","41d71924-826e-4873-a7d3-5a16d198e6f6")
MODEL_ID = os.getenv("MODEL_ID","meta-llama/llama-3-3-70b-instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 8192))
IBM_TIMEOUT = int(os.getenv("IBM_TIMEOUT", 60))
MODEL_ID_HISTORY = os.getenv("MODEL_ID","meta-llama/llama-3-3-70b-instruct")

# logger = logging.getLogger(__name__)
# authenticator = IAMAuthenticator(API_TOKEN_IBM)
MODEL_FROM = os.getenv("MODEL_FROM", "IBM")

if MODEL_FROM=="IBM":
    authenticator = IAMAuthenticator(API_TOKEN_IBM)
else:
    authenticator = lambda *args, **kwargs: ""

# Change to 5 if needed
MEMORY_SIZE = 5
class RedisConnections:

    def __init__(self):
        # Connect to Redis (adjust host/port/db if needed)
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=REDIS_DECODE_RES,
            username = REDIS_USERNAME,
            password = REDIS_PASSWORD
        )

    # Add question/answer/document for a specific user
    def update_memory(self,key, question, answer):
        qa_pair = {'question': question, 'answer': answer}
        self.redis_client.rpush(key, json.dumps(qa_pair))
        self.redis_client.ltrim(key, -MEMORY_SIZE, -1)


    # Get chat history for prompt for a specific user
    def get_chat_history_prompt(self, key, query_enhncement):
        chat_items = self.redis_client.lrange(key, -1, -1)
        is_history=False
        # --- System instructions to control history usage ---
        history_prompt = """
        <|start_header_id|>system<|end_header_id|>
        Use the following conversation history only to understand user intent, references, or ambiguity.
        Do NOT treat history as a source of truth.
        Do NOT copy answers from history.
        Only the current retrieval text should be used to form the final answer.
        <|eot_id|>
        """

        # --- Build conversation history in Meta LLaMA chat format ---
        if query_enhncement:
            for item in chat_items:
                qa_pair = json.loads(item)
                history_prompt += f"""
                    <|start_header_id|>user<|end_header_id|>
                    {qa_pair['question']}
                    <|eot_id|>
                """
                is_history= True
        else:
            for item in chat_items:
                qa_pair = json.loads(item)
                history_prompt += f"""
                    <|start_header_id|>user<|end_header_id|>
                    {qa_pair['question']}
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    {qa_pair['answer']}
                    <|eot_id|>
                """
                is_history= True
        return history_prompt,is_history

    # Clear memory for a user
    def reset_memory(self,key):
        self.redis_client.delete(key)


    def get_token(self):
        """
        Fetch a Bearer token from IBM IAM Authenticator.
        """
        try:
            token = authenticator.token_manager.get_token()
            return "Bearer " + token
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch IBM Cloud token: {e}")
            raise

    def get_history_summary(self,text, query,model_answer):
    # modify the model call functions I just did for the implementation of the VLLM
        
        service = self.get_token()
        # logger.info(
        #     f"\n{'+' * 200}\nSuccessfully obtained authentication token : {ea - sa} seconds where token = {service[-10:]}\n{'+' * 200}\n")

        # API endpoint setup
        url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"


        prompt = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an assistant that summarizes conversation context to clearly reflect what the user asked and how the model responded.
        Your task:
        - You will be given:
        1) A summary of previous conversation history
        2) The current question asked by the user
        3) The assistant’s response to that question
        - Produce a single, coherent, MINIMAL paragraph. The paragraph must explicitly describe the interaction using phrasing such as:
        “the user asked about… and the model responded by…”
        - Combine earlier discussion and the current exchange into a unified narrative.
        - Emphasize the user’s current intent, goals, and how the model’s responses addressed those goals.
        - Maintain chronological flow from earlier context to the current exchange.
        - USER'S CURRENT QUESTION MUST BE EMPHASISED THE MOST AND PLAY THE BIGGEST ROLE IN USER INTENT DESCRIPTION. 
        Strict rules:
        - Do NOT introduce new facts or external knowledge.
        - Do NOT quote the question or answer verbatim.
        - Do NOT use bullet points, lists, or headings.
        - Do NOT include system, developer, or instructional content.
        - Do NOT infer intent beyond what is clearly supported by the provided content.
        - Do NOT treat the summary as factual or authoritative knowledge.
        - Do NOT exclude any detail from the current question previous summaries.
        Formatting rules:
        - Output must be exactly one paragraph.
        - Length should be close to 500 words.
        - Use clear, neutral, professional language.
        - Avoid meta commentary about summarization.
        - Do not mention that this is a summary or that the input was Q&A.
        Purpose:
        This paragraph will be passed to another language model only to convey conversation intent and direction. It must not be used as a knowledge source.
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Previous conversation summary:
        {text}
        Current question asked by the user:
        {query}
        Previous Conversation Summary:
        {model_answer}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>


        """

        # logger.info("Constructing request body with parameters")
        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 500,
                "stop_sequences": ["END"],
                "repetition_penalty": 1.1,
                "temperature": 0.5
            },
            "model_id": MODEL_ID_HISTORY,
            "project_id": PROJECT_ID_IBM,
        }

        # Headers setup
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": service
        }

        # logger.info("Initiating API call to IBM Watson ML with retry mechanism")

        try:
            response = requests.post(url, headers=headers, json=body, timeout=IBM_TIMEOUT)
            et = time.time()
            # logger.info(f"\n{'-' * 200}\nTIME TAKEN BY CALL TO IBM CLOUD : {et - st} seconds\n{'-' * 200}")

        except Exception as e:
            logger.error(f"API call failed with error: {str(e)}")
            return "I am unable to respond right now. Please try again later."

        try:
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            return "I am unable to respond right now. Please try again later."

        try:
            generated_text = data['results'][0]['generated_text']
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract generated text from response: {str(e)}")
            logger.info(f"Response structure: {data}")
            return "I am unable to respond right now. Please try again later."

        logger.info(f"\n{ '_' * 20}\nSummary of history : {generated_text}\n{ '_' * 20}\n")

        return generated_text
