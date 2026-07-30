from fastapi import FastAPI
from llm import generate_answer
from pydantic import BaseModel
from vector_store import search

app = FastAPI(title="Simple RAG App")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    top_documents = search(request.question)
    context = "\n".join(top_documents)
    answer = generate_answer(context, request.question)
    return ChatResponse(answer=answer)
