"""
    this is the augmentation layer 
    - this is the service layer that calls the ES_Con for document search and model_call for text generation
"""
import os
import time
import re
import asyncio
import es_call
import logging
import model_call
import chat_history
from query_enhancement import ibm_model_query
from reranker import rerank_hits
from references import used_references
from attachments import ingest_attachments

es = es_call.EsConnector()
model = model_call.ModelCall()
logger = logging.getLogger(__name__)

model_from = os.getenv("MODEL_FROM", "IBM")

GREETING_RESPONSE = os.getenv("GREETING_RESPONSE", "Hello! How can I help you with your documents today?")
# a message that is only a greeting: hi, hiii, hello there, hey team, good morning ...
GREETING_PATTERN = re.compile(r"(hi+|hey+|hello+|helo|hola|namaste|greetings|good (morning|afternoon|evening|day))( (there|everyone|all|team))?")
ATTACHMENTS_UNREADABLE = "I couldn’t read the attached document, so I can’t answer from it. Please check the file and attach it again."

def is_greeting(query):
    """true when the whole message is a greeting, so there is nothing to search for"""
    words = " ".join(re.sub(r"[^a-z]", " ", query.lower()).split())
    return bool(GREETING_PATTERN.fullmatch(words))

class UniversalDocutalk :
    def __init__(self):
        pass
    async def search_documents_gpt(self,query, username, model_type, answer_type, path,query_id,conversationId,modelName, thinkingRequired, scope, attachments=None):
        # logger.info(f"*-- search_document_function. --*")

        filelist = []
        search_results = []
        if is_greeting(query):
            logger.info("Greeting received --> replying without document search")
            yield [{"response_text": GREETING_RESPONSE}]
            return
        if answer_type not in ["singleDocument", "multiDocument"]:
            logger.warning(f"Unsupported answer type : {answer_type}")
            yield [{"response_text": f"Unsupported answer type: {answer_type}. Supported types: ['singleDocument', 'multiDocument']"}]
        if model_type not in ["NLP", "EXT"]:
            logger.warning(f"Unsupported responses type : {model_type}")
            yield [{"response_text": f"Unsupported responses type: {model_type}. Supported types: ['NLP', 'EXT']"}]

        # a turn with attached files searches only those files, a turn without searches the drive as before
        file_ids = []
        if attachments and answer_type in ["singleDocument", "multiDocument"]:
            start_ingest = time.time()
            file_ids, unread = await asyncio.to_thread(ingest_attachments, attachments, username)
            logger.info(f"Attachments ready in {time.time() - start_ingest:.2f}s --> searching {file_ids}, not read {unread}")
            if not file_ids:
                # the search would find nothing and the answer would blame the question, not the file
                yield [{"response_text": ATTACHMENTS_UNREADABLE}]
                return
            if unread:
                # answering from part of what was attached has to be visible in the answer
                yield [{"response_text": f"<p><em>I couldn’t read {len(unread)} of the {len(attachments)} attached documents, so this answer comes from the rest.</em></p>"}]

        model_answer = ''
        if answer_type == 'singleDocument' :
            history_context_enhance  = es.get_conversation_by_id(conversationId)
            logger.info(f"History for enhancement : {history_context_enhance}")
            ls_query = ibm_model_query(query,model_from,history_context_enhance,answer_type)
            hits = es.search_attachments([ls_query], file_ids) if file_ids else es.search_docs_gpt(ls_query, username, path, scope)
            hits, relevant = rerank_hits(ls_query, hits)
            if hits:
                logger.info("\n\nsingleDocument\n\n")
                score = hits[0].get("_score", 0.0)
                _id = hits[0].get("_id", 0.0)
                file_id = hits[0]["_source"]["fId"]
                page_no = hits[0]["_source"].get("pageNo", "")
                text = hits[0]["_source"].get("text", "")
                filename = hits[0]["_source"].get("fileName", "")
                # logger.info(f"score = {score}\nid = {_id}\n{file_id}\n{page_no}")
                if not relevant:
                    logger.warning("No relevant chunk --> answering from the best fallback chunk")
                search_results.append({"filename": filename, "username": username,"fId": file_id, "page_no": page_no, "score": score, "id":_id , "query_id":query_id, "query":query , "status": 0,"text" : text  })
                if file_ids:
                    # an attached docx is all page 1, so the pages around the best chunk would add nothing to it:
                    # the file's other kept chunks go to the model with it
                    text = " ".join(hit["_source"].get("text", "") for hit in hits if hit["_source"]["fId"] == file_id)
                index = es_call.ATTACHMENT_INDEX if file_ids else es_call.DOCUTALK_INDEX
                combined_text_single_doc = self.above_and_below_page_data(text, int(page_no), file_id, index)
                combined_text_single_doc = f"/n [Filename: {filename} ] /n {combined_text_single_doc} /n"
                # first : fetch the history of past chats
                # history_context,is_history = redis_history.get_chat_history_prompt(conversationId,False)
                # meta_data is sent after the answer, with only the file the answer cites
                # call the model : if it is not first call of the session (context is not empty) then we pass history to model
                if history_context_enhance:
                    logger.info({" Chat History is available "})
                    if model_from == 'IBM' :
                        async for chunk in  model.ibm_cloud(history_context_enhance, combined_text_single_doc, query, model_type,modelName, thinkingRequired, bool(file_ids)):
                            model_answer+=chunk
                            yield [{"response_text": chunk}]
                    elif model_from == 'VLLM':
                        model_answer = model.vllm_model_call(history_context_enhance, combined_text_single_doc, query, model_type)
                # else we pass black history to model
                else:
                    logger.info(" Chat History not available ")
                    if model_from == "IBM":
                        async for chunk in  model.ibm_cloud(" ", combined_text_single_doc, query, model_type,modelName, thinkingRequired, bool(file_ids)):
                            model_answer+=chunk
                            yield [{"response_text": chunk}]
                    elif model_from == 'VLLM':
                        model_answer = model.vllm_model_call(history_context_enhance, combined_text_single_doc, query, model_type)
                logger.info(f"#---model answer {model_answer} ---#")
                # an answer from attached documents has no sources, so no meta_data either
                references = [] if file_ids else used_references(model_answer, search_results, relevant)
                if references:
                    logger.info(f"meta_data sent --> {references}")
                    yield [{"meta_data": references}]
                else:
                    logger.info("meta_data not sent: no reference file for this answer")
                es.store_chat_history(conversationId, query, model_answer)
            else: 
                logger.warning(f"No documents found -->")
                yield [{"response_text": "Thank you for your search! I couldn’t find information on this topic in your uploaded documents. You can try using different keywords or upload a relevant file, and I’ll be happy to help."}]

        elif answer_type == "multiDocument":
            start_es=time.time()
            history_context_enhance= es.get_conversation_by_id(conversationId)
            logger.info(f"Time for taking history from es: {time.time()-start_es}")
            logger.info(f"History for enhancement : {history_context_enhance}")
            ls_query = ibm_model_query(query,model_from,history_context_enhance,answer_type)
            list_query = [m[1] for m in re.findall(r'(["\'])(.*?)\1', ls_query)]
            list_query.append(query)
            hits = es.search_attachments(list_query, file_ids) if file_ids else es.search_docs_gpt_split_query(list_query, username, path, scope)
            # list_query[0] is the rewritten standalone question (or the user's query when no rewrite was found)
            hits, relevant = rerank_hits(list_query[0], hits)
            if hits:
            # logger.info(f"\n\n\n\n\nhits : {hits}\n\n\n\n\n")
            # logger.info(f"time taken for elastic call with split : {time.time() - st}")
                combined_text_multi_doc = ""
                model_answer = ""
                for hit in hits:
                    score = hit.get("_score", 0.0)
                    doc_id = hit.get("_id", "")
                    filename = hit["_source"].get("fileName", "")
                    file_id = hit["_source"].get("fId", "")
                    text = hit["_source"].get("text", "")
                    page_no = hit["_source"].get("pageNo", "")
                    combined_text_multi_doc += f"/n/n/n [FILENAME = {filename} ] /n {text} /n/n/n"

                    # one reference per file, by fId so two different files with the same name are both shown
                    if file_id not in filelist :
                        filelist.append(file_id)
                        search_results.append({"filename": filename, "username": username,"fId": file_id, "page_no": page_no, "score": score, "id":doc_id , "query_id":query_id, "query":query , "status": 0,"text" : text  })

                # logging.info(f"combined_text_multi_doc : {combined_text_multi_doc}")

                # first : fetch the history of past chats
                # meta_data is sent after the answer, with only the files the answer cites
                if not relevant:
                    logger.warning("No relevant chunk --> answering from the best fallback chunks")
                logger.info(f"Files given to the model --> {[result['filename'] for result in search_results]}")

                if history_context_enhance:
                    logger.info(" Chat History is available ")
                    # logger.info(f"\nhistory_context : {history_context}\n\n\n")
                    # changed because UNIVERSAL DOCUTALK had problem with comparison questions
                    if model_type == 'NLP' :
                        model_type = 'NLP_U'  #for UNIVERSAL docutalk
                    elif model_type == "EXT" :
                        model_type = "EXT_U"  #for UNIVERSAL docutalk
                    # logger.info(f"\n{'_'*10}MODEL CALL WITH model_type = {model_type}{'_'*10}\n")
                    if model_from == "IBM":
                        async for chunk in model.ibm_cloud(history_context_enhance, combined_text_multi_doc, query, model_type,modelName,thinkingRequired, bool(file_ids)):
                            model_answer+=chunk
                            yield [{"response_text": chunk}]
                    elif model_from == 'VLLM':
                        model_answer = model.vllm_model_call(history_context_enhance, combined_text_multi_doc, query, model_type,)

                # else we pass blank history to model
                else:
                    logger.info(" Chat History not available ")
                    if model_from == "IBM":
                        async for chunk in model.ibm_cloud(" ", combined_text_multi_doc, query, model_type, modelName, thinkingRequired, bool(file_ids)):
                            model_answer+=chunk
                            yield [{"response_text": chunk}]
                    elif model_from == 'VLLM':
                        model_answer = model.vllm_model_call(history_context_enhance, combined_text_multi_doc, query, model_type)
                # an answer from attached documents has no sources, so no meta_data either
                references = [] if file_ids else used_references(model_answer, search_results, relevant)
                if references:
                    logger.info(f"meta_data sent --> {references}")
                    yield [{"meta_data": references}]
                else:
                    logger.info("meta_data not sent: no reference file for this answer")
                es.store_chat_history(conversationId, query, model_answer)
                logger.info(f"#-- model answer is : {model_answer} --#")
            
            else:
                logger.warning(f"No documents found -->")
                yield [{"response_text": "Thank you for your search! I couldn’t find information on this topic in your uploaded documents. You can try using different keywords or upload a relevant file, and I’ll be happy to help."}]
            
    @staticmethod
    def above_and_below_page_data(text, page_no, file_id, index=es_call.DOCUTALK_INDEX):

        """So this code basically give the full context :
         now text will be =  whole previous page + whole page of chunk + whole next page,

         It helps to get the whole context of the chunk """

        page_no_below = page_no + 1
        below_page_text =es.data_by_page_no(page_no_below, file_id, index)
        # logger.info(f"below_page_text 1 : {below_page_text}")
        # logger.info(f"below_page_text 1 : {below_page_text.keys()}")
        if below_page_text is not None:
            below_page_text = below_page_text['merged_text']
            # logger.info(f"below_page_text 2 : {below_page_text}")
        else:
            below_page_text = ''

        #gets text from previous page
        if page_no != 1:
            page_no_above = page_no - 1
            above_page_text = es.data_by_page_no(page_no_above, file_id, index)
            # logger.info(f"above_page_text : {above_page_text}")

            if above_page_text is not None:
                above_page_text = above_page_text['merged_text']
                # logger.info(f"above_page_text : {above_page_text}")
            else:
                above_page_text = ''
            return above_page_text + text + below_page_text # if chunk is not from the first page

        else:
            page_no_above = page_no + 2
            below_page_text_2 = es.data_by_page_no(page_no_above, file_id, index)

            if below_page_text_2 is not None:
                below_page_text_2 = below_page_text_2['merged_text']
                # logger.info(f"below_page_text_2 : {below_page_text_2}")
            else:
                below_page_text_2 = ''

            return text + below_page_text + below_page_text_2 # if chunk is from the first page



