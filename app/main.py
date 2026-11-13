"""
    this is the main file 
    - api hits here
"""


from fastapi import FastAPI, Body, HTTPException
from typing import List, Union
import uvicorn
import os
import time
import logging
import asyncio
import json
import contextvars
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Must run BEFORE importing service/es_call/doc_ingest — they read their settings at
# import time, so anything loaded later never reaches them. Finds the .env above this
# file. Real environment variables win over it, so a container still overrides.
load_dotenv()

SERVICE_PORT = int(os.getenv("SERVICE_PORT",8989))
GATEWAY_TIME = os.getenv("GATEWAY_TIME",0)

# Must run BEFORE importing service/chat_history/es_call — those modules make
# logger calls (e.g. GraphRAG init) at import time, and any log emitted before
# basicConfig() is silently dropped (INFO) or printed unformatted (WARNING+).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

import service, es_call
from attachments import AttachedDocument, AttachmentBody, attached_documents

app = FastAPI()
logger = logging.getLogger(__name__)

docutalk = service.UniversalDocutalk()
es = es_call.EsConnector()

# the ERROR logs (from any module) of the request being streamed; None outside a stream
request_errors = contextvars.ContextVar("request_errors", default=None)

class RequestErrorCollector(logging.Handler):
    """keeps the ERROR logs of the running request, so stream_chunk can log them with the exact response sent to the backend"""
    def emit(self, record):
        errors = request_errors.get()
        if errors is not None:
            errors.append(f"{record.name} - {record.getMessage()}")

logging.getLogger().addHandler(RequestErrorCollector(logging.ERROR))


"""by the time you uncomment this code, remove the above code because that will become obsolete"""

@app.post("/teamsync/nlp/docutalk/search_documents")
async def universal_docutalk(query,username,modeltype,answer_type,query_id, conversationId,modelName, scope, thinkingRequired: bool, session : bool, path='T_',
                             body: Union[List[AttachedDocument], AttachmentBody, None] = Body(default=None)):
    """ username name passed in redis or history related function is treated as a key for a particular session"""
    logging.info(f"Session Type :: {type(session)}")
    username = username.replace("@", "_")
    # session is a boolean value (if session is false we clear the history of that session from redis server  else docutalk runs and history get stored)

    if session :
        logger.info(f"{'+' * 100} \n BACKEND_FOLDER_PATH : {path} \n {'+' * 100}")
        path = "T" + path.replace("/", "_") if not path.startswith("T_") else path
        logger.info(
            f"{'+' * 10} \n CHANGED_FOLDER_PATH : {path} \n USERNAME : {username} \n MODEL_TYPE : {modeltype} \n ANSWER_TYPE : {answer_type} \n {'+' * 10}")
        try:
            attachments = attached_documents(body)
        except ValueError as e:
            logger.error(f"Request failed before streaming --> sent to backend : HTTP 422 '{e}'")
            raise HTTPException(status_code=422, detail=str(e))
        if attachments:
            logger.info(f"ATTACHMENTS : {attachments}")
        try:
            es.create_conversation_index()
        except Exception:
            logger.error("Request failed before streaming --> sent to backend : HTTP 500 'Internal Server Error'")
            raise
        if scope == "Drive" :
            scope = "username"
        elif scope == "Shared With Others" :
            scope = "shared_by"
        elif scope == "Shared With Me" :
            scope = "shared_with"
        else: 
            scope = "none"
        async def stream_chunk():
            errors, sent = [], []
            request_errors.set(errors)
            try:
                async for chunk in docutalk.search_documents_gpt(query,username,modeltype,answer_type,path,query_id, conversationId, modelName, thinkingRequired, scope, attachments):
                    # logger.info(f"CHUNK: {chunk} ")
                    frame = f"data: {json.dumps(chunk)}\n\n"
                    sent.append(frame)
                    yield   frame
            except Exception as e:
                # the backend gets only the frames already sent, then the connection is cut
                logger.exception(f"Stream failed ({e!r}) --> stream cut off")
                raise
            finally:
                request_errors.set(None)
                if not sent:
                    # e.g. IBM returned a non-200 status: no text and no fallback
                    errors.append("main - nothing was sent to the backend")
                if errors:
                    logger.error(f"Errors in this request --> {errors}")
                    logger.error(f"Exact response sent to backend ({len(sent)} frame(s)) --> {''.join(sent)!r}")
        logger.info(f"{'+'*10} \t\tUniversal DONE\t\t {'+'*10} \n\n")
        if GATEWAY_TIME:
            await asyncio.sleep(int(GATEWAY_TIME))
            
    return StreamingResponse(stream_chunk(),media_type="text/event-stream")




@app.post("/teamsync/nlp/docutalk/close_unv_session")
async def search_documents(username,conversationId, session : bool):
    """ username name passed in redis or history related function is treated as a key for a particular session"""
    # es.delete_conversation(conversationId)
    return "close"
    logging.info(f"Session Type :: {type(session)}")
    key = username.replace("@", "_")
    # session is a boolean value (if session is false we clear the history of that session from redis server  else docutalk runs and history get stored)
    if not session   :
        redis_history.reset_memory(conversationId)
        logger.info(f"UNIVERSAL chat session is closed for {conversationId}")
        result  = "close"
        return result

@app.post("/teamsync/nlp/docutalk/delete_conversation")
async def deleteconvo(conversationId):
    logger.info(f"Deleting conversation with conversation_id :{conversationId}")
    es.delete_conversation(conversationId)
    logger.info(f"Deleted conversation with conversation_id :{conversationId}")
    return "deleted"


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=SERVICE_PORT)
