#------------------------------------Imports-----------------------

from __future__ import annotations

import tempfile
import sqlite3
from typing import Annotated, Any, Dict, Optional, TypedDict
import os
import requests
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS 
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

#load environment variables
load_dotenv()

#-----Create Instance of LLM and Embeddings----------
llm  = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



#--------------this part manages retrievers for PDFs, stored per thread/session-----------

_thread_retrievers: Dict[str, Any] = {}
_thread_metadata: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _thread_retrievers:
        return _thread_retrievers[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id:str, filename:Optional[str] = None) -> dict:
    """
    Build a FAISS retriver for the upoaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the  UI.
    """

    # user uploads pdf -> bytes(stores in memory as bytes) → write to disk → get path → pass to loader
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        #chunks (text) → embeddings (numbers) → FAISS (vector DB) → retriever (search tool)
        #splitting document in priority order
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k":4}
        )

        # saving the retriever + file info for a specific user/session (thread_id)
        _thread_retrievers[str(thread_id)] = retriever
        _thread_metadata[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents":len(docs),
            "chunks": len(chunks)
        }
    finally:
        # The Vector store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

#------------------Tools-----------------------------------------------------------------------------

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error":"No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }
    
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query":query,
        "context": context,
        "metadata": metadata,
        "source_file": _thread_metadata.get(str(thread_id), {}).get("filename")
    }

tools = [search_tool, rag_tool]
llm_with_tools = llm.bind_tools(tools)


#---------------------------------------State---------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


#------------------------------------Nodes-------------------------------------------------

def chat_node(state: ChatState, config=None):
    """
    LLM node that may answer or request a tool call.
    """
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get ("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF , call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search tool when helpful. If no document is available, ask the user "
            "to upload a PDF."  
        )
    )

    messages=[system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)


#----------------------------Checkpointer----------------------------------------

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

#----------------------------Graph----------------------------------------------

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

#---------------------------Helpers---------------------------------------------


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _thread_retrievers

def thread_document_metadata(thread_id: str) -> dict:
    return _thread_metadata.get(str(thread_id), {})
