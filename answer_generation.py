
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

persistent_directory = "db/chroma_db"

# 1. Load the LOCAL embedding model
# Use the same embedding model used to create the vector database.
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 2. Connect to the existing Chroma database
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# 3. Define the query
query = "How much did Microsoft pay to acquire GitHub?"

# 4. Retrieve relevant documents
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# 5. Combine the query and retrieved documents
context = "\n".join(
    f"- {doc.page_content}" for doc in relevant_docs
)

combined_input = f"""
Based on the following documents, answer this question:
{query}

Documents:
{context}

Use only the information in the documents.
If the answer is not present, say:
"I don't have enough information to answer that question
based on the provided documents."
"""

# 6. Initialize the LOCAL Llama 3.2 model
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# 7. Define messages
messages = [
    SystemMessage(
        content="You are a helpful assistant. "
                "Answer questions using only the supplied context."
    ),
    HumanMessage(content=combined_input),
]

# 8. Generate the response locally
result = model.invoke(messages)

print("\n--- Generated Response ---")
print(result.content)
