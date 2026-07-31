from clients.chat import get_chat_client

client = get_chat_client()


def build_prompt(context: str, question: str) -> str:
    return f"""Answer the question using only the context below.
        Context:
        {context}

        Question:
        {question}
    """


def generate_answer(context: str, question: str) -> str:
    prompt = build_prompt(context, question)
    return client.complete([{"role": "user", "content": prompt}])
