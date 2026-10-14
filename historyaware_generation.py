from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)

# Connect to your local Chroma database
persistent_directory = "db/chroma_db"

# Local embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
)

# Set up the local Llama 3.2 model
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)

# Store conversation history
chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite the question using conversation history
    if chat_history:
        messages = [
            SystemMessage(
                content=(
                    "Given the conversation history, rewrite the "
                    "new question as a standalone question that "
                    "can be searched in a document database. "
                    "Return only the rewritten question."
                )
            ),
            *chat_history,
            HumanMessage(
                content=f"New question: {user_question}"
            ),
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()

        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve relevant documents
    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.splitlines()[:2]
        preview = "\n".join(lines)

        print(f"  Doc {i}: {preview}...")

    # Step 3: Build the context from retrieved documents
    context = "\n\n".join(
        f"- {doc.page_content}" for doc in docs
    )
#HERE WERE THE COOKING BEGIN
    combined_input = f"""
Answer the user's question using the documents below.

User question: {user_question}

Retrieved documents:
{context}

Instructions:
- Use only information supported by the documents.
- Consider the conversation history when interpreting the question.
- If the documents do not contain the answer, say:
  "I don't have enough information to answer that question
  based on the provided documents."
- Do not invent facts.
"""

    # Step 4: Generate an answer using local Llama 3.2
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions "
                "using the provided documents and conversation history."
            )
        ),
        *chat_history,
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)
    answer = result.content

    # Step 5: Save this exchange in conversation history
    chat_history.append(
        HumanMessage(content=user_question)
    )
    chat_history.append(
        AIMessage(content=answer)
    )

    print(f"\nAnswer: {answer}")

    return answer


# Simple interactive chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            ask_question(question)
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    start_chat()
