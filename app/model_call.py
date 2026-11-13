"""
    this generates the text from provided prompt+data
"""
import requests
import prompt
import time
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import logging
import os
import httpx
import json
# from dotenv import load_dotenv
# load_dotenv()
logger = logging.getLogger(__name__)

API_TOKEN_IBM = os.getenv("API_TOKEN_IBM")
PROJECT_ID_IBM = os.getenv("PROJECT_ID_IBM")
IBM_MODEL_ID = os.getenv("IBM_MODEL_ID","meta-llama/llama-3-3-70b-instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 8192))
IBM_TIMEOUT = int(os.getenv("IBM_TIMEOUT", 60))
URL = os.getenv("URL", "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation_stream?version=2023-05-29")
STOP_SEQUENCES = [s.strip() for s in (os.getenv("STOP_SEQUENCES","<|eot_id|>") ).split(",") if s.strip()]
# STOP_SEQUENCES = os.getenv("STOP_SEQUENCES", "<|eot_id|>").split(",") if os.getenv("STOP_SEQUENCES") else []  

url_vllm = os.getenv("URL_VLLM","http://172.168.1.221:31766/v1/chat/completions")
vllm_llm_model =os.getenv("VLLM_LLM_MODEL","/models/Llama-3.1-8B-Instruct")
FREQUENCY_PENALTY_NLP=float(os.getenv("FREQUENCY_PENALTY",0.6))
FREQUENCY_PENALTY_EXT=float(os.getenv("FREQUENCY_PENALTY_EXT",0.0))

PRESENCE_PENALTY_NLP=float(os.getenv("PRESENCE_PENALTY_NLP",1))
PRESENCE_PENALTY_EXT=float(os.getenv("PRESENCE_PENALTY_EXT",1))

TEMPERATURE_NLP=float(os.getenv("TEMPERATURE_NLP",0.4))
TEMPERATURE_EXT=float(os.getenv("TEMPERATURE_EXT",0.1))

TOP_P_NLP=float(os.getenv("TOP_P_NLP",0.9))
TOP_P_EXT=float(os.getenv("TOP_P_EXT",1.0))

REPETITION_PENALTY_NLP=float(os.getenv("REPETITION_PENALTY_NLP",1.05))
DECODING_METHOD_NLP=os.getenv("DECODING_METHOD_NLP","greedy")

LLAMA_3_3_70B= os.getenv("LLAMA_3_3_70B","meta-llama/llama-3-3-70b-instruct")
MIXTRAL_8x_7B= os.getenv("MIXTRAL_8x_7B","mistralai/mistral-medium-2505")
LLAMA_4_SCOUT_17B = os.getenv("LLAMA_4_SCOUT_17B","meta-llama/llama-4-maverick-17b-128e-instruct-fp8")
LLAMA_3_3_8B = os.getenv("LLAMA_3_3_8B","meta-llama/llama-3-3-8b-instruct")

# answer sent to the backend when the IBM call fails
FALLBACK_ANSWER = "I am unable to respond right now. Please try again later."

format_prompt = prompt.CallPrompt()
class ModelCall:
    def __init__(self):
        # Create an IAM authenticator using the token from env
        model_from_ = os.getenv("MODEL_FROM", "IBM")
        if model_from_ == "IBM":
            self.authenticator = IAMAuthenticator(API_TOKEN_IBM)
            _ = self.get_token()
        else:
            self.authenticator = lambda *args, **kwargs: ""
        # Warm-up (forces token fetch once on startup)
    def get_token(self) -> str:
        """
        Fetch a Bearer token from IBM IAM Authenticator.
        """
        try:
            token = self.authenticator.token_manager.get_token()
            return "Bearer " + token
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch IBM Cloud token: {e}")
            raise

    @staticmethod
    async def send_post_request(url, headers, body):
        """
        Sends an HTTP POST request to the IBM Cloud API and yields streamed responses.
        """
        try:
            async with httpx.AsyncClient(timeout=IBM_TIMEOUT) as client:
                # POST request (must await)
                async with client.stream("POST", url, headers=headers, json=body) as response:

                    # async iteration over streaming lines
                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        decoded_line = line.strip()
                        if decoded_line.startswith("data: "):
                            try:
                                chunk_data = json.loads(decoded_line[len("data: "):])
                                if "results" in chunk_data and chunk_data["results"]:
                                    result = chunk_data["results"][0]

                                    text_chunk = result.get("generated_text", "")
                                    if text_chunk:
                                        yield {"type": "text", "content": text_chunk}

                                    moderations = result.get("moderations", {}).get("hap", [])
                                    if moderations:
                                        flagged_words = [m["word"] for m in moderations if "word" in m]
                                        if flagged_words:
                                            yield {"type": "moderation", "flagged": flagged_words}
                            except json.JSONDecodeError:
                                continue

                if response.status_code != 200:
                    logger.error(f"IBM Cloud API returned status {response.status_code}")

        except httpx.RequestError as e:
            logger.error(f"HTTP request to IBM Cloud failed: {e}")
            raise

    async def ibm_cloud(self, history: str, text: str, query: str, prompt_type: str, modelName: str, thinkingRequired: bool, attached: bool = False):
        # logger.info(f"Starting ibm_cloud with prompt_type={prompt_type}")

        # Build prompt via template 
        base_prompt = format_prompt.ibm_prompt_formating(prompt_type, thinkingRequired, modelName, attached)

        # NOTE: avoid logging the full prompt in production (contains user data)
        prompt_text = base_prompt.format(text=text, query=query, history=history)

        # Auth
        if modelName == "Llama-3.3-70B":
            IBM_MODEL_ID=LLAMA_3_3_70B
        elif modelName == "Mixtral-8x7B":
            IBM_MODEL_ID=MIXTRAL_8x_7B
        elif modelName == "Llama-4-Scout-17B":
            IBM_MODEL_ID=LLAMA_4_SCOUT_17B
        elif modelName == "Llama3-8B":
            IBM_MODEL_ID=LLAMA_3_3_8B
        service_token = self.get_token()
        # logger.info(f"Obtained IBM auth token in {time.time() - st:.3f}s")
        if prompt_type == "NLP" or "NLP_U":
            # Request body (all env-driven)
            body = {
                "input": prompt_text,
                "project_id": PROJECT_ID_IBM,     # <-- now from env, not hardcoded
                "model_id": IBM_MODEL_ID,         # <-- from env
                "parameters": {
                    "frequency_penalty": FREQUENCY_PENALTY_NLP,
                    "max_new_tokens": MAX_NEW_TOKENS,  # <-- from env
                    "presence_penalty": PRESENCE_PENALTY_NLP,
                    "temperature": TEMPERATURE_NLP,
                    "top_p": TOP_P_NLP,
                    "decoding_method": DECODING_METHOD_NLP,
                    "repetition_penalty": REPETITION_PENALTY_NLP,
                    "stop_sequences": STOP_SEQUENCES
                },
            }

            headers = {
                "Accept": "text/event_stream",
                "Content-Type": "application/json",
                "Authorization": service_token,
            }

            try:
                call_st = time.time()
                async for chunk in self.send_post_request(url=URL, headers=headers, body=body):
                    if chunk["type"] == "text":
                        # print(chunk["content"], end="------", flush=True)
                        yield chunk["content"]
                    elif chunk["type"] == "moderation":
                        logger.warning(f"Flagged words detected: {chunk['flagged']}")
                        yield f"\n[Content moderated: {', '.join(chunk['flagged'])}]"
                # logger.info(f"IBM call time: {time.time() - call_st:.3f}s")
            except Exception as e:
                # return a friendly message to the caller
                logger.error(f"IBM model call failed ({e!r}) --> sending to backend : {FALLBACK_ANSWER}")
                yield FALLBACK_ANSWER
        

        elif prompt_type == "EXT" or "EXT_U":
            # Request body (all env-driven)

            body = {
                "input": prompt_text,
                "project_id": PROJECT_ID_IBM,     # <-- now from env, not hardcoded
                "model_id": IBM_MODEL_ID,         # <-- from env
                "parameters": {
                    "frequency_penalty": FREQUENCY_PENALTY_EXT,
                    "max_new_tokens": MAX_NEW_TOKENS,  # <-- from env
                    "presence_penalty": PRESENCE_PENALTY_EXT,
                    "temperature": TEMPERATURE_EXT,
                    "top_p": TOP_P_EXT,
                    "stop_sequences": STOP_SEQUENCES
                },
            }

            headers = {
                "Accept": "text/event_stream",
                "Content-Type": "application/json",
                "Authorization": service_token,
            }

            try:
                call_st = time.time()
                async for chunk in self.send_post_request(url=URL, headers=headers, body=body):
                    if chunk["type"] == "text":
                        # print(chunk["content"], end="------", flush=True)
                        yield chunk["content"]
                    elif chunk["type"] == "moderation":
                        logger.warning(f"Flagged words detected: {chunk['flagged']}")
                        yield f"\n[Content moderated: {', '.join(chunk['flagged'])}]"
                # logger.info(f"IBM call time: {time.time() - call_st:.3f}s")
            except Exception as e:
                # return a friendly message to the caller
                logger.error(f"IBM model call failed ({e!r}) --> sending to backend : {FALLBACK_ANSWER}")
                yield FALLBACK_ANSWER

    @staticmethod
    def vllm_model_call(history, text, query, model_type):
        # Improvise the prompt, I made the basic prompt to start
        prompt = format_prompt.vllm_prompt_formating(history, text, model_type)
        # logger.info(f"Prompt : {prompt}")
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": vllm_llm_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        }
        try:
            logger.info("inside try of vllm")
            response = requests.post(url=url_vllm, headers=headers, json=payload)
            data = response.json()
            # logger.info(f"JSON response of Vllm model : {data}")
            final_response = data["choices"][0]["message"]["content"]
            logger.info(f"Final response  : {final_response}")
            return final_response
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

