"""this file works on the Universal Docutalk"""

import requests
import prompt
import time
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import logging
import os

logger = logging.getLogger(__name__)

API_TOKEN_IBM = os.getenv("API_TOKEN_IBM")
PROJECT_ID_IBM = os.getenv("PROJECT_ID_IBM")
MODEL_ID = os.getenv("MODEL_ID","meta-llama/llama-3-3-70b-instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 8192))
IBM_TIMEOUT = int(os.getenv("IBM_TIMEOUT", 60))
#---- Deprecated -> got error : message': "Model 'meta-llama/llama-3-1-8b-instruct' is not supported",

URL_VLLM = os.getenv("URL_VLLM","http://172.168.1.221:31766/v1/chat/completions")
VLLM_LLM_MODEL =os.getenv("VLLM_LLM_MODEL","/models/Llama-3.1-8B-Instruct")

MODEL_FROM = os.getenv("MODEL_FROM", "IBM")

if MODEL_FROM=="IBM":
    authenticator = IAMAuthenticator(API_TOKEN_IBM)
else:
    authenticator = lambda *args, **kwargs: ""

def get_token():
    """
    Fetch a Bearer token from IBM IAM Authenticator.
    """
    try:
        token = authenticator.token_manager.get_token()
        return "Bearer " + token
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch IBM Cloud token: {e}")
        raise


def ibm_model_query(query , model_form, history_enhance ,answer_type):
    # modify the model call functions I just did for the implementation of the VLLM
    if model_form == "IBM":
        # Authentication
        sa = time.time()
        # logger.info("Retrieving authentication token")
        service = get_token()
        ea = time.time()
        # logger.info(
        #     f"\n{'+' * 200}\nSuccessfully obtained authentication token : {ea - sa} seconds where token = {service[-10:]}\n{'+' * 200}\n")

        # API endpoint setup
        url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

        if answer_type == "multiDocument":
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You resolve references in a user's question. You never answer it, never split it, and never improve it.

Choose one of two actions:

KEEP — the question already stands on its own. Copy it exactly, character for character.
RESOLVE — the question cannot be understood on its own because it points at something outside itself. Rewrite it as one self-contained question by substituting the subject from the chat history. Change nothing else: keep the user's own words, word order and level of detail.

RESOLVE only when one of these is true:
1. The question contains a reference with no subject inside the question itself: it, its, this, that, these, those, they, them, their, he, she, the same, the above, the former, the latter.
2. The question is a fragment that only makes sense as a follow-up: "and for 2024?", "what about the second one", "more details", "why?", "continue".

Everything else is KEEP. In particular:
- A question that names its own subject is KEEP, even when the chat history is about that same subject.
- A long, complex or multi-part question is KEEP. Never break it apart.
- A broad but self-contained question ("what are the main points?") is KEEP.
- If the chat history is empty, or nothing in it explains the reference, KEEP.

Output rules:
- Output exactly one Python list holding exactly one double-quoted string: ["..."]
- Never output two or more elements. Never split on "and", on commas, or on any other separator.
- Use double quotes, never single quotes.
- No explanation, no labels, no code fences, no repetition of these instructions.
- Write END after the list.

Examples:

Chat History: The user asked about the leave policy for contract employees.
Current Query: How many days does it allow?
["How many days does the leave policy for contract employees allow?"] END

Chat History: The user asked about Jaguar aircraft engine specifications.
Current Query: Compare the specifications of Jaguar and Mikoyan.
["Compare the specifications of Jaguar and Mikoyan."] END

Chat History:
Current Query: What is the penalty for late submission of the annual report?
["What is the penalty for late submission of the annual report?"] END

Chat History: The user asked about the 2023 travel reimbursement limits.
Current Query: and for 2024?
["What are the travel reimbursement limits for 2024?"] END

Chat History: The user asked about the employee onboarding checklist.
Current Query: What documents are required for vendor registration?
["What documents are required for vendor registration?"] END
<|eot_id|><|start_header_id|>user<|end_header_id|>
Chat History:
{history_enhance}
Current Query:
{query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        elif answer_type=="singleDocument":
            prompt=f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You rewrite or enhance a user's query using the provided chat history as context.
            Purpose:
            - Resolve vague references like "it", "this", "its disadvantages", "explain more", etc.
            - Infer the correct subject only from the chat history.
            - Produce one clear, self-contained question.
            Rules:
            - Use chat history strictly as reference to identify the subject.
            - Do NOT introduce new topics not present in history.
            - Keep the question concise and explicit.
            - Output must be a single string.
            - Do NOT add explanations, prefixes, or formatting.
            - Do NOT return a list.
            Examples:
            User Query: Explain its disadvantages
            Chat History: Discussion about Redis as a cache
            Assistant: "Explain the disadvantages of using Redis as a cache"
            User Query: How does it work?
            Chat History: Elasticsearch vector search explanation
            Assistant: "How does Elasticsearch vector search work?"
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Chat History:
            {history_enhance}
            Current Query:
            {query}
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
            "model_id": MODEL_ID,
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
            st = time.time()
            response = requests.post(url, headers=headers, json=body, timeout=IBM_TIMEOUT)
            et = time.time()
            # logger.info(f"\n{'-' * 200}\nTIME TAKEN BY CALL TO IBM CLOUD : {et - st} seconds\n{'-' * 200}")

        except Exception as e:
            logger.error(f"API call failed with error: {str(e)}")
            return "I am unable to respond right now. Please try again later."

        try:
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return "I am unable to respond right now. Please try again later."

        try:
            generated_text = data['results'][0]['generated_text']
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract generated text from response: {str(e)}")
            logger.info(f"Response structure: {data}")
            return "I am unable to respond right now. Please try again later."

        logger.info(f"\n{ '_' * 20}\nmulti query : {generated_text}\n{ '_' * 20}\n")


        return generated_text

    elif model_form == "VLLM" :
        prompt = f""" You simplify or split given Query in multiple sentence.
            Instruction to create sentence :
            - Simply the query in small chunks (because data is available in different files)
            - use the keywords from User Query that will make the search faster
            - create multiple query
            - minimum sentence = 1
            - maximum sentence = 3
            - generate response in list format ['','','']
        
            Here's the example :
            Query = What is the difference between features of abc and xyz ?
            Assistant = [' features of abc', ' features of xyz']
            
            Query = Compare the specifications of Jaguar and Mikoyan.
            Assistant = [' specifications of Jaguar', ' specifications of Mikoyan', 'jaguar and Mikoyan']
        
            Query = {query}
            """
        # logger.info(f"Prompt : {prompt}")
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": VLLM_LLM_MODEL,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        }
        try:
            logger.info("inside try of vllm")
            response = requests.post(url=URL_VLLM, headers=headers, json=payload)
            data = response.json()
            logger.info(f"JSON response of Vllm model : {data}")
            final_response = data["choices"][0]["message"]["content"]
            logger.info(f"Final response  : {final_response}")
            return final_response
        except Exception as e:
            logger.error(f"Unexpected error: {e}")


# print(ibm_model("What is your name ?"))


