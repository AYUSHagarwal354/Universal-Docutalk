"""
    it calls elasticsearch to search the data based on the query 
"""

from elasticsearch import Elasticsearch
import os
import logging
import re
import string
from datetime import datetime,timezone


logger = logging.getLogger(__name__)
# http://172.168.1.215:30612/

# 172.168.1.217:31987
ES_HOST =os.getenv("ES_HOST","elasticsearch-teamsync.apps.lab.ocp.lan")
ES_PORT = int(os.getenv("ES_PORT",80))
ELASTIC_PASSWORD =os.getenv("ELASTIC_PASSWORD","admin123")
ELASTIC_USER =os.getenv("ELASTIC_USER","elastic")
RESPONSE_SIZE_FOR_FILEID=int(os.environ.get("RESPONSE_SIZE_FOR_FILEID",20))
DEFAULT_SIZE_UNIVERSAL_SINGLE=int(os.environ.get("DEFAULT_SIZE_UNIVERSAL_SINGLE",100))
SCHEME=os.environ.get("SCHEME","http")
DOCUTALK_INDEX=os.getenv("DOCUTALK_INDEX","teamsync_v1") #FIXME
ATTACHMENT_INDEX=os.getenv("ATTACHMENT_INDEX","docutalk") # documents attached to a chat turn, kept out of the drive corpus
DEFAULT_SIZE_UNIVERSAL_MULTI = int(os.environ.get("DEFAULT_SIZE_UNIVERSAL_MULTI",30))
MODEL_ID = os.getenv("model_id_es", ".multilingual-e5-small_linux-x86_64")
ES_REQUEST_TIMEOUT: int = int(os.environ.get("ES_REQUEST_TIMEOUT", 30))
RETRIEVER_MODE = os.getenv("RETRIEVER_MODE","hybrid_rrf") # other option is vector
MAX_EXPANSIONS = int(os.getenv("MAX_EXPANSIONS",100))

NUM_CANDIDATES_SINGLE= int(os.getenv("NUM_CANDIDATES_SINGLE",100))
NUM_CANDIDATES_MULTIPLE= int(os.getenv("NUM_CANDIDATES_MULTIPLE",100))

RANK_WINDOW_SIZE_SINGLE= int(os.getenv("RANK_WINDOW_SIZE_SINGLE",100))
RANK_CONSTANT_SINGLE =int(os.getenv("RANK_CONSTANT_SINGLE",60))

RANK_WINDOW_SIZE_MULTIPLE= int(os.getenv("RANK_WINDOW_SIZE_MULTIPLE",100))
RANK_CONSTANT_MULTIPLE= int(os.getenv("RANK_CONSTANT_MULTIPLE",60))
HISTORY_INDEX=os.getenv("HISTORY_INDEX","history")
HISTORY_SIZE = int(os.getenv("HISTORY_SIZE", 3))
NUM_BUCKETS = int(os.getenv("NUM_BUCKETS", 3))
# a misspelt word ("gosh" for "ghosh") matches nothing in BM25, so the words of the query that are in at most
# FUZZY_RARE_MAX of the user's chunks are also matched fuzzily (1 edit for 3-5 letters, 2 for longer words, first letter fixed)
FUZZY_RARE_WORDS = os.getenv("FUZZY_RARE_WORDS", "true").lower() == "true"
FUZZY_RARE_MAX = int(os.getenv("FUZZY_RARE_MAX", 1))
VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", 0.82))

# common english words that say nothing about a document's content. they are left out of the keyword (BM25) search only:
# the vector search and the reranker still get the full question
KEYWORD_STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom whose why
will with would you your yours yourself yourselves
""".split())
KEYWORD_MIN_SHOULD_MATCH = os.getenv("KEYWORD_MIN_SHOULD_MATCH", "75%")   # share of the keywords a chunk must contain


def keyword_text(query):
    """the question without stop words and single characters ("D." in "D. Ghosh"), for the keyword search; "" when nothing is left"""
    # an initial written without a space ("d.gosh") is split off, so it is dropped like "d. gosh"; "U.S." is kept
    query = re.sub(r"\b([A-Za-z])\.(?=[A-Za-z]{2})", r"\1 ", query)
    words = (word.strip(string.punctuation + "“”‘’") for word in query.split())
    return " ".join(word for word in words if len(word) > 1 and word.lower() not in KEYWORD_STOP_WORDS)

class EsConnector:
    
    def __init__(self) -> None:
        self.es_client = None
        self.connect()

    def connect(self):
        self.es_client = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme':SCHEME}],basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),request_timeout=ES_REQUEST_TIMEOUT)
        # logger.info({self.es_client.ping()})
        
    
#__________________________________________________________________________________

    def search_docs_gpt(self, query, username, path, scope):
        username_list=[]
        for i in range(NUM_BUCKETS):
            username_list.append(f"{username}_{i}")
        if RETRIEVER_MODE == "vector":
            return self.search_docs_gpt_by_vectoronly(query,username,path, username_list,scope)
        else:
            return self.search_docs_gpt_by_hybridrrf(username,query,path,username_list, scope)
        
#__________________________________________________________________________________

    def search_docs_gpt_split_query(self, list_query, username, path, scope):
       username_list=[]
       for i in range(NUM_BUCKETS):
            username_list.append(f"{username}_{i}")
       if RETRIEVER_MODE== "vector":
           return self.search_docs_gpt_split_query_vectoronly( list_query, username, path, username_list, scope)
       else:
           return self.search_docs_gpt_split_query_hybridrrf(list_query, username,path, username_list, scope)
    
#-----__________------------_____________________________-----------------------------
    
    def search_docs_gpt_by_vectoronly(self, query, username,path, username_list, scope):
        es_query=[]
        if scope == "username" or scope == "shared_by":
            es_query.append({
                        "knn": {
                            "field": "text_embedding",
                            "k": DEFAULT_SIZE_UNIVERSAL_SINGLE,
                            "num_candidates": NUM_CANDIDATES_SINGLE,
                            "query_vector_builder": {
                                "text_embedding": {
                                    "model_id": MODEL_ID,
                                    "model_text": query
                                }
                            },
                            "filter": [
                                {"term": {"username": username}},
                                {"term": {"in_trash": False}},
                                {"match_phrase_prefix": {"path.text": {
                                    "query":path,
                                    "max_expansions": MAX_EXPANSIONS}
                                    }
                                }
                            ]
                        }
                    })
            route = ",".join(username_list)
        elif scope == "shared_by":
            es_query.append({
                        "knn": {
                            "field": "text_embedding",
                            "k": DEFAULT_SIZE_UNIVERSAL_SINGLE,
                            "num_candidates": NUM_CANDIDATES_SINGLE,
                            "query_vector_builder": {
                                "text_embedding": {
                                    "model_id": MODEL_ID,
                                    "model_text": query
                                }
                            },
                            "filter": [
                                {"term": {"username": username}},
                                {"term": {"in_trash": False}},
                                {"exists": {
                                    "field": "shared_with"
                                    }
                                }
                            ]
                        }
                    })
            route = ",".join(username_list)
        elif scope =="shared_with":
            es_query.append({
                        "knn": {
                            "field": "text_embedding",
                            "k": DEFAULT_SIZE_UNIVERSAL_SINGLE,
                            "num_candidates": NUM_CANDIDATES_SINGLE,
                            "query_vector_builder": {
                                "text_embedding": {
                                    "model_id": MODEL_ID,
                                    "model_text": query
                                }
                            },
                            "filter": [
                                {"term": {"shared_with": username}},
                                {"term": {"in_trash": False}}
                            ]
                        }
                    })
            route = None
        else: 
            return []
        response = self.es_client.search(
            index=DOCUTALK_INDEX,
            routing = route,
            body={
                "size": DEFAULT_SIZE_UNIVERSAL_SINGLE,
                "sort": [{"_score":{"order":"desc"}}],
                "query": es_query,
                "_source": [
                    "text", "pageNo", "fId", "username", "fileName"
                ]
            }
        )
        # logger.info(f"Hits received :{response['hits']['hits']}")
        return response['hits']['hits']

    def rare_word_retriever(self, query, filters, route):
        """a retriever that matches the query's rare words fuzzily (see FUZZY_RARE_MAX), None when there is none or the check fails"""
        if not FUZZY_RARE_WORDS or not query:
            return None
        try:
            tokens = self.es_client.indices.analyze(index=DOCUTALK_INDEX, body={"field": "text", "text": query})["tokens"]
            words = list(dict.fromkeys(token["token"] for token in tokens))
            if not words:
                return None
            named = {f"t{i}": {"term": {"text": word}} for i, word in enumerate(words)}
            response = self.es_client.search(index=DOCUTALK_INDEX, routing=route, body={"size": 0, "track_total_hits": False,
                "aggs": {"user": {"filter": {"bool": {"filter": filters}}, "aggs": {"t": {"filters": {"filters": named}}}}}})
            buckets = response["aggregations"]["user"]["t"]["buckets"]
            rare = [word for i, word in enumerate(words) if buckets[f"t{i}"]["doc_count"] <= FUZZY_RARE_MAX]
        except Exception as e:
            logger.warning(f"Rare word check failed, no fuzzy matching: {e}")
            return None
        if not rare:
            return None
        logger.info(f"Fuzzy matching the rare words {rare}")
        return {"standard": {
            "query": {"bool": {"should": [
                {"match": {"text": {"query": word, "fuzziness": "AUTO", "prefix_length": 1, "max_expansions": 50}}} for word in rare]}},
            "filter": filters}}

    def search_docs_gpt_split_query_hybridrrf(self, list_query, username, path, username_list, scope):

        retriever_list=[]
        if scope == "username" :
            user_filter = [
                {"term": {"username": username}},
                {"term": {"in_trash": False}},
                {"match_phrase_prefix": {"path.text": {"query": path, "max_expansions": MAX_EXPANSIONS}}},
            ]
            # the keyword searches (BM25 and fuzzy) get the question without stop words and single letters
            rare_word_retrievers = {query: self.rare_word_retriever(keyword_text(query), user_filter, ",".join(username_list)) for query in dict.fromkeys(list_query)}
            for query in list_query:
                retriever_list.append(
                {
                "knn": {
                    "field": "text_embedding",
                    "k": DEFAULT_SIZE_UNIVERSAL_MULTI,
                    "num_candidates": NUM_CANDIDATES_MULTIPLE,
                    "similarity": VECTOR_SIMILARITY_THRESHOLD,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": MODEL_ID,
                            "model_text": query
                        }
                    },
                    "filter": [
                        {"term": {"username": username}},
                        {"term": {"in_trash": False}},
                        {"match_phrase_prefix": {"path.text": {
                            "query":path,
                            "max_expansions": MAX_EXPANSIONS}
                            }
                        }
                    ]
                }
            })
                # skipped when no keyword is left (the vector and fuzzy searches still run)
                keywords = keyword_text(query)
                if keywords:
                    retriever_list.append({"standard": {
                        "query": {"multi_match": {"query": keywords, "fields": ["text"], "minimum_should_match": KEYWORD_MIN_SHOULD_MATCH}},
                        "min_score": 2,
                        "filter": [
                            {"term": {"username": username}},
                            {"term": {"in_trash": False}},
                            {"match_phrase_prefix": {"path.text": {
                                "query":path,
                                "max_expansions": MAX_EXPANSIONS}
                                }
                            }
                        ]
                        }
                    })
                if rare_word_retrievers[query]:
                    retriever_list.append(rare_word_retrievers[query])
            route = ",".join(username_list)
        elif scope =="shared_with":
            for query in list_query:
                retriever_list.append(
                {
                "knn": {
                    "field": "text_embedding",
                    "k": DEFAULT_SIZE_UNIVERSAL_MULTI,
                    "num_candidates": NUM_CANDIDATES_MULTIPLE,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": MODEL_ID,
                            "model_text": query
                        }
                    },
                    "filter": [
                        {"term": {"shared_with": username}},
                        {"term": {"in_trash": False}}
                    ]
                }
            })
                keywords = keyword_text(query)
                if keywords:
                    retriever_list.append({"standard": {
                        "query": {"multi_match": {"query": keywords, "fields": ["text"], "minimum_should_match": KEYWORD_MIN_SHOULD_MATCH}},
                        "filter": [
                            {"term": {"shared_with": username}},
                            {"term": {"in_trash": False}}
                        ]
                        }
                    })
            route = None
            logger.info("Shared with route, no routing applied")
        elif scope == "shared_by":
            for query in list_query:
                retriever_list.append(
                {
                "knn": {
                    "field": "text_embedding",
                    "k": DEFAULT_SIZE_UNIVERSAL_MULTI,
                    "num_candidates": NUM_CANDIDATES_MULTIPLE,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": MODEL_ID,
                            "model_text": query
                        }
                    },
                    "filter": [
                        {"term": {"username": username}},
                        {"term": {"in_trash": False}},
                        {"exists": {
                            "field": "shared_with"
                            }
                        }
                    ]
                }
            })
                keywords = keyword_text(query)
                if keywords:
                    retriever_list.append({"standard": {
                        "query": {"multi_match": {"query": keywords, "fields": ["text"], "minimum_should_match": KEYWORD_MIN_SHOULD_MATCH}},
                        "filter": [
                            {"term": {"username": username}},
                            {"term": {"in_trash": False}},
                            {"exists": {
                                "field": "shared_with"
                                }
                            }
                        ]
                        }
                    })
            route = ",".join(username_list)
        else: 
            return []
        
        logger.info(f"\n\n{'='*20}MULTI DOC QUERY = {retriever_list}\n{'='*20}")

        response = self.es_client.search(
            index=DOCUTALK_INDEX,
            routing= route,
            body={
                "size": DEFAULT_SIZE_UNIVERSAL_MULTI,
                "_source": [
                    "text", "pageNo", "fId", "username", "fileName"
                ],
                "retriever":{
                    "rrf":{
                        "retrievers":retriever_list,
                        "rank_window_size":RANK_WINDOW_SIZE_MULTIPLE,
                        "rank_constant":RANK_CONSTANT_MULTIPLE
                    }
                }
            }
        )
        return response['hits']['hits']

    

    def search_docs_gpt_split_query_vectoronly(self, list_query, username, path, username_list, scope):

        if scope == "username" :
            query = {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "field": "text_embedding",
                                "k": DEFAULT_SIZE_UNIVERSAL_MULTI,
                                "num_candidates": NUM_CANDIDATES_MULTIPLE,
                                "query_vector_builder": {
                                    "text_embedding": {
                                        "model_id": MODEL_ID,
                                        "model_text": query_
                                    }
                                },
                                "filter": [
                                    {"term": {"username": username}},
                                    {"term": {"in_trash": False}},
                                    {"match_phrase_prefix": {"path.text": {
                                        "query":path,
                                        "max_expansions": MAX_EXPANSIONS}
                                        }
                                        }
                                ]
                            }
                        } for query_ in list_query
                    ]
                }
            }
            route = ",".join(username_list)
        elif scope == "shared_by":
            query = {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "field": "text_embedding",
                                "k": DEFAULT_SIZE_UNIVERSAL_MULTI,
                                "num_candidates": NUM_CANDIDATES_MULTIPLE,
                                "query_vector_builder": {
                                    "text_embedding": {
                                        "model_id": MODEL_ID,
                                        "model_text": query_
                                    }
                                },
                                "filter": [
                                    {"term": {"username": username}},
                                    {"term": {"in_trash": False}},
                                    {"exists": {
                                        "field": "shared_with"
                                        }
                                    }
                                ]
                            }
                        } for query_ in list_query
                    ]
                }
            }
            route = ",".join(username_list)

        elif scope == "shared_with":
            query = {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "field": "text_embedding",
                                "k": DEFAULT_SIZE_UNIVERSAL_MULTI,
                                "num_candidates": NUM_CANDIDATES_MULTIPLE,
                                "query_vector_builder": {
                                    "text_embedding": {
                                        "model_id": MODEL_ID,
                                        "model_text": query_
                                    }
                                },
                                "filter": [
                                {"term": {"shared_with": username}},
                                {"term": {"in_trash": False}}
                                ]
                            }
                        } for query_ in list_query
                    ]
                }
            }
            route = None
        else: 
            return []
        logger.info(f"\n\n{'='*20}MULTI DOC QUERY = {query}\n{'='*20}")

        response = self.es_client.search(
            index=DOCUTALK_INDEX,
            routing= route,
            body={
                "size": DEFAULT_SIZE_UNIVERSAL_MULTI,
                "sort": [{"_score":{"order":"desc"}}],
                "query": query,
                "_source": [
                    "text", "pageNo", "fId", "username", "fileName"
                ]
            }
        )
        return response['hits']['hits']

    def data_by_page_no(self,page_no, fid, index=DOCUTALK_INDEX):
        # logger.info(f"Index selected : {index}")
        match = self.es_client.search(
            index=index,
            body={
                "size": 15,
                "sort":[{"para":{"order":"asc"}}],
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"pageNo": page_no}},
                            {"match": {"fId": fid}}
                        ]
                    }
                },
                "script_fields": {
                    "text": {
                        "script": "params['_source']['text']"
                    }
                }
            }
        )
        if match['hits']['hits']:
            all_text = " ".join(hit["fields"]["text"][0] for hit in match['hits']['hits'])
            return {"merged_text": all_text}


    def search_attachments(self, list_query, fids):
        """
        hybrid RRF search (BM25 + vector) over the attached files only.
        the fIds decide what is searched, so username, path and scope play no part, and there is no similarity or
        keyword threshold: the user picked these files, so the reranker decides which chunks are relevant.
        """
        filters = [{"terms": {"fId": fids}}, {"term": {"in_trash": False}}]
        retriever_list = []
        for query in dict.fromkeys(list_query):
            retriever_list.append({
                "knn": {
                    "field": "text_embedding",
                    "k": DEFAULT_SIZE_UNIVERSAL_MULTI,
                    "num_candidates": NUM_CANDIDATES_MULTIPLE,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": MODEL_ID,
                            "model_text": query
                        }
                    },
                    "filter": filters
                }
            })
            # the full question when no keyword is left, so rrf always has both searches to merge
            retriever_list.append({"standard": {
                "query": {"multi_match": {"query": keyword_text(query) or query, "fields": ["text"]}},
                "filter": filters
            }})

        response = self.es_client.search(
            index=ATTACHMENT_INDEX,
            body={
                "size": DEFAULT_SIZE_UNIVERSAL_MULTI,
                "_source": [
                    "text", "pageNo", "fId", "username", "fileName"
                ],
                "retriever": {
                    "rrf": {
                        "retrievers": retriever_list,
                        "rank_window_size": RANK_WINDOW_SIZE_MULTIPLE,
                        "rank_constant": RANK_CONSTANT_MULTIPLE
                    }
                }
            }
        )
        return response['hits']['hits']

    def search_docs_gpt_by_hybridrrf(self, username, query, path, username_list, scope):
        """
        New RRF-based hybrid retriever (BM25 + vector search).
        """
        # logger.info(f"Username : {username}\t path : {path}\t Index selected : {DOCUTALK_INDEX}, function : search_docs_gpt_by_hybridrrf ")
        es_query=[]
        # the keyword search gets the question without stop words and single letters, and is skipped when none is left
        keywords = keyword_text(query)
        if scope == "username" :
            if keywords:
                es_query.append({
                                "standard": {
                                    "query": {
                                        "match": {
                                            "text": {"query": keywords, "minimum_should_match": KEYWORD_MIN_SHOULD_MATCH}
                                        }
                                    },
                                    "min_score": 2,
                                    "filter": [
                                        {"term": {"username": username}},
                                        {"term": {"in_trash": False}},
                                        {"match_phrase_prefix": {"path.text": {
                                            "query":path,
                                            "max_expansions": MAX_EXPANSIONS}
                                            }
                                        }
                                    ]
                                }
                            })
            es_query.append({
                                "knn": {
                                    "field": "text_embedding",
                                    "k": DEFAULT_SIZE_UNIVERSAL_SINGLE,
                                    "num_candidates": NUM_CANDIDATES_SINGLE,
                                    "similarity": 0.75,
                                    "query_vector_builder": {
                                        "text_embedding": {
                                            "model_id": MODEL_ID,
                                            "model_text": query
                                        }
                                    },
                                    "filter": [
                                        {"term": {"username": username}},
                                        {"term": {"in_trash": False}},
                                        {"match_phrase_prefix": {"path.text": {
                                            "query":path,
                                            "max_expansions": MAX_EXPANSIONS}
                                            }
                                        }
                                    ]
                                }
                            }
                        )
            route= ",".join(username_list)

        elif scope == "shared_by":
            if keywords:
                es_query.append({
                                "standard": {
                                    "query": {
                                        "match": {
                                            "text": {"query": keywords, "minimum_should_match": KEYWORD_MIN_SHOULD_MATCH}
                                        }
                                    },
                                    "filter": [
                                        {"term": {"username": username}},
                                        {"term": {"in_trash": False}},
                                        {"exists": {
                                            "field": "shared_with"
                                            }
                                        }
                                    ]
                                }
                            })
            es_query.append({
                                "knn": {
                                    "field": "text_embedding",
                                    "k": DEFAULT_SIZE_UNIVERSAL_SINGLE,
                                    "num_candidates": NUM_CANDIDATES_SINGLE,
                                    "query_vector_builder": {
                                        "text_embedding": {
                                            "model_id": MODEL_ID,
                                            "model_text": query
                                        }
                                    },
                                    "filter": [
                                        {"term": {"username": username}},
                                        {"term": {"in_trash": False}},
                                        {"exists": {
                                            "field": "shared_with"
                                            }
                                        }
                                    ]
                                }
                            }
                        )
            route= ",".join(username_list)

        elif scope == "shared_with":
            if keywords:
                es_query.append({
                                "standard": {
                                    "query": {
                                        "match": {
                                            "text": {"query": keywords, "minimum_should_match": KEYWORD_MIN_SHOULD_MATCH}
                                        }
                                    },
                                    "filter": [
                                        {"term": {"shared_with": username}},
                                    ]
                                }
                            })
            es_query.append({
                                "knn": {
                                    "field": "text_embedding",
                                    "k": DEFAULT_SIZE_UNIVERSAL_SINGLE,
                                    "num_candidates": NUM_CANDIDATES_SINGLE,
                                    "query_vector_builder": {
                                        "text_embedding": {
                                            "model_id": MODEL_ID,
                                            "model_text": query
                                        }
                                    },
                                    "filter": [
                                        {"term": {"shared_with": username}}
                                    ]
                                }
                            }
                        )
            route= None 
        else:
            return []
        response = self.es_client.search(
            index=DOCUTALK_INDEX,
            routing= route,
            body={
                "retriever": {
                    "rrf": {
                        "retrievers": es_query,
                        "rank_window_size": RANK_WINDOW_SIZE_SINGLE,
                        "rank_constant": RANK_CONSTANT_SINGLE
                    }
                },
                "size": DEFAULT_SIZE_UNIVERSAL_SINGLE,
                "_source": [
                    "text", "pageNo", "fId", "username", "fileName"
                ]
            }
        )

        return response['hits']['hits']
    
    def create_conversation_index(self):
        """
        Create an Elasticsearch index if it does not exist
        with mappings to store conversation ID and chat history.
        """

        try:
            if self.es_client.indices.exists(index=HISTORY_INDEX):
                logger.info(f"Index '{HISTORY_INDEX}' already exists.")
                return

            mapping = {
                "mappings": {
                    "properties": {
                        "conversation_id": {
                            "type": "keyword"
                        },
                        "qa_pairs": {
                            "type": "nested",
                            "properties": {
                                "question": { "type": "text" },
                                "answer": { "type": "text" },
                                "created_at": { "type": "date" }
                            }
                        }
                    }
                }
            }

            self.es_client.indices.create(
                index=HISTORY_INDEX,
                body=mapping
            )

            logger.info(f"Index '{HISTORY_INDEX}' created successfully.")

        except Exception as e:
            logging.error(f"Failed to create index '{HISTORY_INDEX}': {e}")
            raise


    def get_conversation_by_id(self, conversation_id):
        """
        Fetch last N QA pairs by slicing in Python (not in Elasticsearch).
        """

        try:
            query = {
                "size": 1,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "conversation_id": conversation_id
                                }
                            }
                        ]
                    }
                }
            }

            response = self.es_client.search(
                index=HISTORY_INDEX,
                body=query
            )

            hits = response["hits"]["hits"]
            if not hits:
                return None

            qa_pairs = hits[0]["_source"].get("qa_pairs", [])

            last_qa_pairs = qa_pairs[-HISTORY_SIZE:]

            qa_list = []
            for qa in last_qa_pairs:
                qa_list.append(
                    f"Question: {qa['question']}\nAnswer: {qa['answer']}"
                )

            return "\n\n".join(qa_list)

        except Exception as e:
            logger.error(f"Error searching conversation_id '{conversation_id}': {e}")
            return None
    
    def store_chat_history(self, conversation_id, question, answer):
        """
        Append a question-answer pair to qa_pairs for a conversation.
        Creates the document if it does not exist.
        """

        try:
            qa_entry = {
                "question": question,
                "answer": answer,
                "created_at" : datetime.now(timezone.utc)
            }

            self.es_client.update(
                index=HISTORY_INDEX,
                id=conversation_id,
                body={
                    "script": {
                        "source": """
                            if (ctx._source.qa_pairs == null) {
                                ctx._source.qa_pairs = [];
                            }
                            ctx._source.qa_pairs.add(params.qa);
                        """,
                        "params": {
                            "qa": qa_entry
                        }
                    },
                    "upsert": {
                        "conversation_id": conversation_id,
                        "qa_pairs": [qa_entry]
                    }
                },
                refresh=True
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store chat history: {e}")
            return False
        
    def delete_conversation(
        self,
        conversation_id: str
    ) -> bool:
        """
        Delete a conversation document from Elasticsearch.
        """

        try:
            self.es_client.delete(
                index=HISTORY_INDEX,
                id=conversation_id,
                refresh=True
            )
            return True

        except Exception as e:
            print(f"Failed to delete conversation: {e}")
            return False

    

    
