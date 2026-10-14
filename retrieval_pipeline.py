from langchain_chroma import Chroma # cuz i can host it locally
import ollama
from langchain_ollama import OllamaEmbeddings
persistent_directory = "db/chroma_db"  # Directory to store the vector database

 # remarqie: we need to work with the same emb model as we used in ingestion pipeline to create the vector store
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)



db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model, collection_metadata={'hnsw:space': 'cosine'})

"** search for relevant docs **"

#query = "Quels sont les produits de Nvidia?"
query = "Quelle est la capital de la France?"


retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5}, score_threshold=0.3)  # ONLY RETURN CHUNKS WITH SCORE ABOVE 0.3

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")