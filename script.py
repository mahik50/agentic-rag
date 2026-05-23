import os
import warnings
import operator
import hashlib
from typing import Literal, Annotated, List
from time import sleep

import requests
import chromadb
from dotenv import load_dotenv
from urllib3.exceptions import InsecureRequestWarning
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

import nest_asyncio
nest_asyncio.apply()

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

load_dotenv() 


LLM_MODEL           = "gpt-5.3-chat"
EMBEDDING_LLM_MODEL = "text-embedding-3-small"
LLM_TEMPERATURE     = 1
NO_PROXY_URL        = ".visa.com"
CHROMA_DB_PATH      = "./chroma_db"
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200
CONTEXT_LIMIT       = 100_000   
TOP_K               = 5


class LlmModel:
    def __init__(self) -> None:
        os.environ["no_proxy"] = NO_PROXY_URL
        self.base_url          = os.getenv("GENAI_BASE_URL")
        self.application_name  = os.getenv("GENAI_PLATFORM_APPLICATION_NAME")
        self.access_token      = os.getenv("GENAI_ACCESS_TOKEN")

    def _post(self, message_list: list) -> str:
        payload = {
            "model_name":       LLM_MODEL,
            "application_name": self.application_name,
            "query":            message_list,
            "customized_params": {
                "temperature": LLM_TEMPERATURE,
                "stream":      False,
            },
        }
        headers = {
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        resp = requests.post(
            f"{self.base_url}/genai-api/v1/queries/chat",
            headers=headers,
            json=payload,
            verify=False,
        )
        resp.raise_for_status()
        result = resp.json()

        if "response" in result:
            return result["response"]
        if "full_model_response" in result:
            choices = result["full_model_response"].get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        if "choices" in result and result["choices"]:
            return result["choices"][0].get("message", {}).get("content", "")
        if "content" in result:
            return result["content"]
        if "text" in result:
            return result["text"]

        raise ValueError(f"Unexpected response format. Keys: {list(result.keys())}")

    def invoke(self, query: str, system_prompt: str = "You are a helpful assistant.") -> str:
        message_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ]
        return self._post(message_list)

    def invoke_with_history(
        self,
        messages: List[BaseMessage],
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        message_list = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if isinstance(msg, HumanMessage):
                message_list.append({"role": "user",      "content": msg.content})
            elif isinstance(msg, AIMessage):
                message_list.append({"role": "assistant", "content": msg.content})
        return self._post(message_list)



class EmbeddingModel:
    def __init__(self) -> None:
        os.environ["no_proxy"] = NO_PROXY_URL
        self.base_url          = os.getenv("GENAI_BASE_URL")
        self.application_name  = os.getenv("GENAI_PLATFORM_APPLICATION_NAME")
        self.access_token      = os.getenv("GENAI_ACCESS_TOKEN")

    def embed(self, query: str) -> list:
        payload = {
            "model_name": EMBEDDING_LLM_MODEL,
            "user_context": {
                "application_name": self.application_name,
                "end_user":         "udhiman",
            },
            "query":             query,
            "customized_params": {},
        }
        headers = {
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        resp = requests.post(
            f"{self.base_url}/genai-api/v1/queries/embedding",
            headers=headers,
            json=payload,
            verify=False,
        )
        resp.raise_for_status()
        result = resp.json()
        return result["full_model_response"]["data"][0]["embedding"]



llm            = LlmModel()
embeddingModel = EmbeddingModel()



def _url_to_collection_name(url: str) -> str:
    digest = hashlib.md5(url.encode()).hexdigest()[:16]
    return f"url_{digest}"


def _html_extractor(html: str) -> str:
    return " ".join(Soup(html, "html.parser").get_text().split())


def load_url(url: str, max_depth: int = 1, progress_callback=None) -> chromadb.Collection:
    collection_name = _url_to_collection_name(url)
    client          = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        if progress_callback:
            progress_callback(" URL already indexed — loading from cache.")
        return client.get_collection(collection_name)

    if progress_callback:
        progress_callback(" Scraping the URL…")

    loader = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=_html_extractor,
        use_async=True,
        ssl=False,
    )
    docs = loader.load()

    if not docs:
        raise ValueError(f"No content could be scraped from: {url}")

    if progress_callback:
        progress_callback(f" Scraped {len(docs)} page(s). Splitting into chunks…")

    
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks    = splitter.split_documents(docs)
    doc_texts = [c.page_content for c in chunks]
    metadatas = [c.metadata     for c in chunks]
    ids       = [f"doc_{i}"     for i in range(len(chunks))]

    if progress_callback:
        progress_callback(f" {len(chunks)} chunks created. Embedding…")

    embeddings = []
    for i, text in enumerate(doc_texts):
        embeddings.append(embeddingModel.embed(text))
        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(f" Embedded {i + 1}/{len(doc_texts)} chunks…")

    if progress_callback:
        progress_callback(" Storing in vector database…")

    collection = client.create_collection(collection_name)
    collection.add(
        ids=ids,
        documents=doc_texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    if progress_callback:
        progress_callback(f" Indexed {len(chunks)} chunks successfully.")

    return collection


def router(query: str) -> Literal["summary", "section", "exhaustive"]:
    system_prompt = """
        You are a query classifier for a web-page Q&A chatbot.

        The user has provided a URL/webpage, and is now asking questions about its content.
        Your sole job is to classify the user's query into exactly ONE of three retrieval strategies:

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ROUTE 1 → summary
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Use when the user wants a high-level overview or synthesis of the ENTIRE page.
        Triggers: "summarize", "overview", "what is this page about", "main idea",
                "key points", "gist", "tldr", "what does this link contain",
                "what is this about", "brief description", "high-level"

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ROUTE 2 → section
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Use when the user asks a SPECIFIC, LOCALIZED question answerable from a few
        relevant chunks via semantic search (top-k retrieval).
        Triggers: specific facts, definitions, dates, names, prices, steps, clauses,
                "what does section X say", "who is", "when did", "how to", "explain X",
                "what is the policy on", "find the part about"

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ROUTE 3 → exhaustive
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Use when the user wants COMPLETE extraction across the ENTIRE page — every
        instance, all occurrences, nothing missed. Requires scanning all content.
        Triggers: "all", "every", "complete list", "full list", "each", "entire",
                "all mentions of", "every time", "list all", "extract all",
                "comprehensive", "nothing missed", "thorough"

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        DECISION RULES (in priority order):
        1. If the query contains "all", "every", "complete", "full list", "each" → exhaustive
        2. If the query asks for overview/summary/gist of the whole page → summary
        3. If the query is specific and localized → section
        4. If ambiguous between summary and section → prefer section for factual questions, summary for broad questions
        5. If ambiguous between exhaustive and summary → prefer exhaustive when completeness is implied

        OUTPUT FORMAT:
        - Return ONLY one word: summary, section, or exhaustive
        - No punctuation, no explanation, no extra text
        - Lowercase only
    """


    result = llm.invoke(query=query, system_prompt=system_prompt).strip().lower()
    if result not in ("summary", "section", "exhaustive"):
        result = "section"  
    return result



def summary_retriever(query: str, collection) -> str:
    SUMMARY_SYSTEM_PROMPT = """
        You are a helpful assistant that summarizes web page content.

        You will be given the full text content of a webpage (possibly split into chunks).
        Your job is to produce a clear, well-structured summary that covers:
        - The main topic and purpose of the page
        - Key points, findings, or information presented
        - Important details, sections, or themes

        Guidelines:
        - Be concise but comprehensive
        - Use bullet points or short paragraphs for clarity
        - Do not hallucinate — only use information from the provided content
        - If the user has a specific focus in their query, emphasize that aspect
    """
    MAP_SYSTEM_PROMPT = """
        You are a summarization assistant.
        Summarize the following chunk of webpage content in 3-5 sentences.
        Capture the key points only. Be concise.
    """
    REDUCE_SYSTEM_PROMPT = """
        You are a summarization assistant.
        You will receive multiple partial summaries of different sections of a webpage.
        Combine them into one coherent, well-structured final summary.
        Eliminate redundancy. Preserve all important information.
        Focus on: main topic, key points, important details.
    """

    results   = collection.get(include=["documents"])
    all_docs  = results["documents"]
    full_text = "\n\n".join(all_docs)

    if len(full_text) <= CONTEXT_LIMIT:
        user_message = f"User query: {query}\n\nPage content:\n{full_text}"
        return llm.invoke(query=user_message, system_prompt=SUMMARY_SYSTEM_PROMPT)
    else:
        chunk_summaries = [
            llm.invoke(query=chunk, system_prompt=MAP_SYSTEM_PROMPT)
            for chunk in all_docs
        ]
        combined   = "\n\n---\n\n".join(chunk_summaries)
        user_message = f"User query: {query}\n\nPartial summaries:\n{combined}"
        return llm.invoke(query=user_message, system_prompt=REDUCE_SYSTEM_PROMPT)



def section_retriever(query: str, collection, top_k: int = TOP_K) -> str:
    SECTION_SYSTEM_PROMPT = """
        You are a helpful assistant answering questions about a webpage.

        You will be given:
        1. A user question
        2. Relevant excerpts retrieved from the webpage

        Your job:
        - Answer the question using ONLY the provided excerpts
        - Be specific and direct
        - Quote or reference the relevant part if helpful
        - If the answer is not found in the excerpts, say:
          "I couldn't find specific information about that in the retrieved sections."
        - Do not hallucinate or use outside knowledge
    """
    query_embedding = embeddingModel.embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    context = ""

    for docs in results['documents'][0]:
        context += "\n\n-----\n\n"
        context += docs


    return context

    



def exhaustive_retriever(query: str, collection) -> str:
    
    EXTRACT_SYSTEM_PROMPT = """
        You are a precise information extractor.

        You will be given:
        1. A user query
        2. A single chunk of webpage content

        Your job:
        - Read the chunk carefully
        - Extract ONLY the parts that are directly relevant to the query
        - If the chunk contains relevant information, return just those parts verbatim or paraphrased
        - If the chunk contains NO relevant information, respond with exactly: "NONE"
        - Do not add commentary, headers, or explanation — just the extracted content or "NONE"
    """
    MERGE_SYSTEM_PROMPT = """
        You are a synthesis assistant.

        You will be given multiple extracted pieces of information from different
        sections of a webpage, all relevant to a user's query.

        Your job:
        - Merge all the extracted pieces into one comprehensive, well-structured answer
        - Remove duplicates and redundant information
        - Organize logically (use bullet points, numbered lists, or sections as appropriate)
        - Preserve all unique information — nothing should be lost
        - Do not hallucinate — only use the provided extracted content
    """
    results  = collection.get(include=["documents", "metadatas"])
    all_docs = results["documents"]
    all_meta = results["metadatas"]

    extracted_parts = []
    for i, (chunk, meta) in enumerate(zip(all_docs, all_meta)):
        user_message = f"Query: {query}\n\nChunk content:\n{chunk}"
        extraction   = llm.invoke(query=user_message, system_prompt=EXTRACT_SYSTEM_PROMPT).strip()
        if extraction and extraction.upper() != "NONE":
            source = meta.get("source", f"chunk_{i}")
            extracted_parts.append(f"[From chunk {i + 1} | {source}]\n{extraction}")

    if not extracted_parts:
        return "No relevant information found across the entire page for your query."

    combined_extractions = "\n\n---\n\n".join(extracted_parts)
    merge_message        = f"Query: {query}\n\nExtracted parts from all chunks:\n{combined_extractions}"
    return llm.invoke(query=merge_message, system_prompt=MERGE_SYSTEM_PROMPT)



class AgentState(TypedDict):
    messages:   Annotated[List[BaseMessage], operator.add]  
    context:    str                                          
    route:      str                                          
    collection: object        
    refined_query: str   
    retries: int                            




def router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content

    if(len(state['refined_query']) > 0):
        last_message = state['refined_query']

    route        = router(last_message)

    print('Router Node', router)


    return {"route": route, "messages": [], "context": state.get("context", ""), "collection": state["collection"], "refined_query": "", 'retries': state['retries']}


def summary_node(state: AgentState) -> dict:
    print('Summary Node')
    query   = state["messages"][-1].content
    context = summary_retriever(query, state["collection"])
    return {"context": context, "messages": [], "route": state["route"], "collection": state["collection"], "refined_query": "", 'retries': state['retries']}


def section_node(state: AgentState) -> dict:
    print('Section Node')
    query   = state["messages"][-1].content
    context = section_retriever(query, state["collection"])
    return {"context": context, "messages": [], "route": state["route"], "collection": state["collection"], "refined_query": "", 'retries': state['retries']}


def exhaustive_node(state: AgentState) -> dict:
    print('Exhaustive Node')
    query   = state["messages"][-1].content
    context = exhaustive_retriever(query, state["collection"])
    return {"context": context, "messages": [], "route": state["route"], "collection": state["collection"], "refined_query": "", 'retries': state['retries']}


def generate_node(state: AgentState) -> dict:
    print('Generate Node')
    GENERATE_SYSTEM_PROMPT = """
        You are a helpful assistant answering questions about a webpage.

        The following context has been retrieved from the webpage the user is asking about.
        Use it as your primary source of truth when answering.

        RETRIEVED CONTEXT:
        {context}

        Guidelines:
        - Answer using the retrieved context above
        - You have access to the full conversation history — use it to handle follow-up questions
        - If the context doesn't contain the answer, say so honestly
        - Be clear, concise, and accurate
    """.format(context=state["context"])

    answer = llm.invoke_with_history(
        messages=state["messages"],
        system_prompt=GENERATE_SYSTEM_PROMPT,
    )

    return {
        "messages":   [AIMessage(content=answer)],
        "context":    state["context"],
        "route":      "",
        "collection": state["collection"],
        "refined_query": "",
        "retries": 0
    }



def route_decision(state: AgentState) -> str:
    return state["route"]



def self_correct(state: AgentState) -> Literal["sufficient", "insufficient"]:
    
    print('Self Correct Node')
    if(state['retries'] > 0):
        return {'route': 'sufficient'}
    

    CONTEXT_CHECK_SYSTEM_PROMPT = """
        You are a context relevance evaluator for a web-page Q&A system.

        You will be given:
        1. A user query
        2. Retrieved context chunks from a vector database

        Your job is to determine whether the retrieved context contains SUFFICIENT information
        to answer the user's query accurately and completely.

        Evaluation criteria:
        - SUFFICIENT: The context directly contains the facts, details, or information needed
        to answer the query. Even a partial but meaningful answer is possible.
        - INSUFFICIENT: The context is off-topic, too vague, missing key details, or completely
        unrelated to what the user is asking.

        OUTPUT FORMAT:
        - Return ONLY one word: sufficient or insufficient
        - No punctuation, no explanation, no extra text
        - Lowercase only

        Examples:
        - Query: "What is the return policy?" | Context contains return policy details → sufficient
        - Query: "Who is the CEO?" | Context talks about product features only → insufficient
        - Query: "How do I install the package?" | Context has installation steps → sufficient
        - Query: "What are the pricing tiers?" | Context has unrelated blog content → insufficient

    """


    query=state['messages'][-1].content
    context=state['context']

    prompt=f"User Query:{query} \n\n Context: {context}"


    


    result=llm.invoke(query=prompt, system_prompt=CONTEXT_CHECK_SYSTEM_PROMPT).strip().lower()

    print('Query Refinement', result)
    if result not in ("sufficient", "insufficient"):
        result="insufficient"

    return {'route': result}



def refine(state: AgentState):
    print('Refine Node')
    QUERY_REWRITE_SYSTEM_PROMPT = """
        You are a search query optimizer for a web-page Q&A system.

        The user asked a question, but the initial vector database search returned
        context that was NOT sufficient to answer it.

        Your job is to rewrite the user's query into a better search query that will
        retrieve more relevant chunks from the vector database using semantic search.

        You are also provided with message history.

        Rewriting strategies:
        - Break down complex questions into simpler, more focused search terms
        - Use different vocabulary or synonyms that might match the document's language
        - Make implicit concepts explicit (e.g., "how much does it cost?" → "pricing cost fee")
        - Remove conversational filler and focus on the core information need
        - If the query is a follow-up question, make it self-contained

        OUTPUT FORMAT:
        - Return ONLY the rewritten query as a plain string
        - No explanation, no prefix like "Rewritten query:", no quotes
        - Keep it concise — 1-2 sentences maximum

        Examples:
        - Original: "what does it cost?" → Rewritten: "pricing plans cost fee subscription"
        - Original: "tell me more about that" → Rewritten: "detailed explanation features benefits"
        - Original: "is it safe to use?" → Rewritten: "security safety privacy data protection"
        - Original: "how do I get started?" → Rewritten: "installation setup getting started quickstart guide"
    """
    
    
    query = state['messages'][-1].content
    
    context = state['context']

    prompt = f"User Query: {query} \n\n Context: {context}"

    temp_messages = state['messages']
    temp_messages.append(HumanMessage(content=prompt))


    refined_query=llm.invoke_with_history(messages=temp_messages, system_prompt=QUERY_REWRITE_SYSTEM_PROMPT)


    return {"route": "", "messages": [], "context": "", "collection": state["collection"], "refined_query": refined_query, "retries": 1}




def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router",     router_node)
    graph.add_node("summary",    summary_node)
    graph.add_node("section",    section_node)
    graph.add_node("exhaustive", exhaustive_node)
    graph.add_node("generate",   generate_node)
    graph.add_node("self_correct", self_correct)
    graph.add_node("refine", refine)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {"summary": "summary", "section": "section", "exhaustive": "exhaustive"},
    )

    graph.add_edge("summary",    "self_correct")
    graph.add_edge("section",    "self_correct")
    graph.add_edge("exhaustive", "self_correct")

    graph.add_conditional_edges(
        "self_correct",
        lambda state: state['route'],
        {"sufficient": "generate", "insufficient": "refine"},
    )

    graph.add_edge("refine", "router")



    graph.add_edge("generate",   END)

    return graph.compile()



class Conversation():
    def __init__(self, url: str):
        self.app = build_graph()
        self.history = []
        self.collection = load_url(url)

    def chat(self, query: str) -> str :
        self.history = self.history + [HumanMessage(content=query)]

        result = self.app.invoke({
            "messages":   self.history,
            "context":    "",
            "route":      "",
            "refined_query": "",
            "collection": self.collection,
            "retries": 0
        })

        ai_reply = result["messages"][-1].content
        self.history = result['messages']

        return ai_reply


# chatbot=Conversation(url="https://reference.langchain.com/python/langgraph/graph/state/StateGraph")

# while True:
#     res=int(input('Enter 1 to exit or 2 to type the query'))
#     if(res == 1):
#         break
#     else:
#         query=input('Enter your query')
#         print(chatbot.chat(query))