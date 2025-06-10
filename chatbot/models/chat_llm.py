from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict

from chatbot.vector_db.document import Document


class State(TypedDict):
    context: List[Document]
    messages: List[dict]


class ChatbotLLM():
    def __init__(self, model_name: str = "mistral:latest", temperature: float = 0.8, num_predict: int = 256, base_url: str = "http://127.0.0.1:11434/"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
            base_url=base_url,
        )

        with open("./lecture-chatbot/chatbot/data/templates/chat_template.txt", "r") as file:
            self.chat_prompt = file.read()

        self.graph = self.init_agent_graph()
    
    def prompt(self, message: str) -> str:
        return self.invoke(
            self.chat_prompt.format(message=message)
        )
    
    def init_agent_graph(self) -> StateGraph:
        graph = StateGraph(State)

        def retrieve(state: State) -> State:
            """
            Retrieve relevant documents based on the current context.
            """
            #TODO: Implement the retrieval logic

            return state
        
        def generate_answer(state: State) -> str:
            """
            Generate an answer based on the retrieved documents and user message.
            """
            context = state["context"]
            messages = state["messages"]
            
            if not context:
                #TODO: implement larger context search logic
                True
            
            # Combine context and messages to form the input for the LLM
            with open("./lecture-chatbot/chatbot/data/templates/generate_answer.txt", "r") as file:
                answer_template = file.read()

            input_text = answer_template.format(
                context="\n".join(doc.content for doc in context),
                messages="\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            )

            return {
                "messages": messages + [self.llm.invoke(input_text)]
            }
        
        graph.add_node("retrieve", retrieve)
        graph.add_node("generate_answer", generate_answer)
        graph.add_edge("retrieve", "generate_answer")
        graph.add_edge(START, "retrieve")
        graph.add_edge("generate_answer", END)

        return graph.compile()
    
    def invoke(self, message: str) -> str:
        """
        Invoke the LLM with the given message and return the response.
        """
        initial_state: State = {
            "context": [],  
            "messages": [{"role": "user", "content": message}] 
        }
    
        return self.graph.invoke(initial_state)

