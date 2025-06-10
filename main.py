from typing import Union

from fastapi import FastAPI

import sys

sys.path.append("./lecture-chatbot")  # Adjust the path to include the parent directory

from chatbot.models.chat_llm import ChatbotLLM

app = FastAPI()

# Initialize the chatbot with the Mistral model running on Ollama server
chatbot = ChatbotLLM(model_name="mistral:latest", temperature=0.8, num_predict=256, base_url="http://127.0.0.1:11434/")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/chat/infer")
def chat_infer(prompt: str):
    return chatbot.invoke(message=prompt)