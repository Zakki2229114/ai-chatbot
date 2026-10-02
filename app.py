import os
import gradio as gr
from langchain_groq import ChatGroq

# Read API key from environment variable (NOT hardcoded)
groq_api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-120b",
    temperature=0
)

def chat_with_llm(message, history):
    result = llm.invoke(message)
    return result.content

demo = gr.ChatInterface(
    fn=chat_with_llm,
    title="My AI Chat",
    description="Type anything and get an answer"
)

# Critical: Render expects the app to listen on a specific port
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )