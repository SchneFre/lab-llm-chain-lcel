#!/usr/bin/env python
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# Load environment variables from .env
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Make sure the key is set for LangChain/OpenAI
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

class TranslationRequest(BaseModel):
    text: str
    language: str

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

# Pass the API key to ChatOpenAI
model = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
parser = StrOutputParser()
chain = prompt_template | model | parser

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
