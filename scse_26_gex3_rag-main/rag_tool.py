from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool

from policy_loader import load_policy_documents


# Load the supplied university policy file.
documents = load_policy_documents()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="university_support_agent",
    embedding_function=embeddings,
)
vector_store.add_documents(documents)


@tool
def search_university_knowledge(question: str) -> str:
    """Search university documentation for IT policies and instructions."""
    results = vector_store.similarity_search(question, k=2)

    retrieved = []
    for document in results:
        source = document.metadata["source"]
        retrieved.append(f"SOURCE: {source}\n{document.page_content}")

    return "\n\n".join(retrieved)
