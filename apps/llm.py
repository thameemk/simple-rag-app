import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(context: str, question: str) -> str:
    return f"""Answer the question using only the context below.
        Context:
        {context}

        Question:
        {question}
    """


def generate_answer(context: str, question: str) -> str:
    prompt = build_prompt(context, question)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
