import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage , SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres  import PostgresSaver
from psycopg import connect
from psycopg.rows import RowFactory
from typing import TypedDict, List, Dict, Sequence, Annotated
from pydantic import BaseModel, Field
from rag import hybrid_search

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("INVALID API KEY")

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)

from typing import TypedDict
from pydantic import BaseModel, Field


class CriticResult(BaseModel):
    approved: bool
    weaknesses: list[str]


class QueryList(BaseModel):
    queries: list[str] = Field(description="3 to 5 short search queries")


class AgentState(TypedDict, total=False):
    resume_text: str         
    current_resume: str    
    search_queries: list[str]
    guidelines: str           
    weaknesses: list[str]     
    approved: bool          
    retry_count: int         
    final_note: str