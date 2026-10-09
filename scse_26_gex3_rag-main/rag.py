from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from policy_loader import load_policy_documents


# Load documents and create the vector store.
documents = load_policy_documents()
print(f"Loaded {len(documents)} documents.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="university_support",
    embedding_function=embeddings,
)
vector_store.add_documents(documents)

# Retrieve the two most relevant documents.
question = "What exact phone number should I call for the university IT Service Desk?"
results = vector_store.similarity_search(question, k=2)

print("\nRETRIEVED DOCUMENTS")
for document in results:
    print("\n--------------------")
    print("Source:", document.metadata["source"])
    print(document.page_content)

context = "\n\n".join(document.page_content for document in results)

# Answer using the retrieved documentation.
model = ChatOllama(model="qwen3:0.6b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a university IT support assistant.
Answer the question using only the supplied university documentation.
If the documentation does not contain the answer, say that you do not know
because the requested information is not specified.
Do not invent phone numbers or university policies.
Do not give unrelated advice. Keep the answer concise.""",
    ),
    (
        "human",
        "UNIVERSITY DOCUMENTATION:\n{context}\n\nQUESTION:\n{question}",
    ),
])

rag_chain = prompt | model
response = rag_chain.invoke({"context": context, "question": question})

print("\nFINAL ANSWER")
print(response.content)
