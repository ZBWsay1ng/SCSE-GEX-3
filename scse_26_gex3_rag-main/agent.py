from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from rag_tool import search_university_knowledge


model = ChatOllama(model="qwen3:0.6b", temperature=0)

agent = create_agent(
    model=model,
    tools=[search_university_knowledge],
    system_prompt="""You are a university IT support assistant.
Use search_university_knowledge before answering questions about university
IT policies, procedures, or contact information.
Answer using only the information returned by the tool.
If the requested information is missing, say that you do not know because
the provided documentation does not specify it.
Do not invent phone numbers or university policies.
Do not give unrelated advice. Keep the answer concise.""",
)

question = "What exact phone number should I call for the university IT Service Desk?"
result = agent.invoke({
    "messages": [{"role": "user", "content": question}]
})

print("\nFINAL ANSWER")
print(result["messages"][-1].content)

print("\nFULL AGENT TRACE")
for message in result["messages"]:
    print("\n--------------------")
    print(type(message).__name__)
    print(message)
