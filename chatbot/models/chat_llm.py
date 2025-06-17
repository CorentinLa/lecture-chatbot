from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import List, TypedDict

from chatbot.vector_db.document import Document
from chatbot.vector_db.vectordb import VectorDB
from chatbot.models.embedding import EmbeddingModel


class State(TypedDict):
    context: List[Document]
    messages: List[dict]
    user: int


class ChatbotLLM():
    def __init__(self, model_name: str = "mistral:latest", temperature: float = 0.3, num_predict: int = 512, base_url: str = "http://127.0.0.1:11434/",
                 embedding_model: EmbeddingModel = None, rag_db: VectorDB = None, rag_result_limit: int = 5):
        
        if embedding_model is None:
            raise ValueError("Embedding model must be provided.")
        if rag_db is None:
            raise ValueError("Vector database must be provided.")
        
        self.embedding_model = embedding_model
        self.rag_db = rag_db
        self.rag_result_limit = rag_result_limit

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
            query = self.embedding_model.embed_query(state["messages"][-1]["embedding_query"])
            
            if not hasattr(state, "context"): state["context"] = [] 

            results = self.rag_db.collection.query(
                query_embeddings=[query],
                n_results=self.rag_result_limit,
                where={
                    "user": state["user"]
                }
            )

            for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]): # 0 because we only query one vector
                if distance < 0.4:
                    state["context"].append({
                        "content": doc,
                        "metadata": metadata
                    }
                    )

            return state
        
        def retrieve_from_all_users(state: State) -> StateGraph:
            """
            Retrieve relevant documents from all users.
            """
            query = self.embedding_model.embed_query(state["messages"][-1]["embedding_query"])
            if not hasattr(state, "context"): state["context"] = [] 


            results = self.rag_db.collection.query(
                query_embeddings=[query],
                n_results=self.rag_result_limit,
            )

            for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                if distance < 0.4:
                    state["context"].append({
                        "content": doc,
                        "metadata": metadata
                    }
                    )
            return state
        
        def no_document_found(state: State) -> StateGraph:
            """
            Handle the case where no relevant documents are found.
            """
            state["messages"].append({
                "role": "assistant",
                "content": "No relevant documents found. Please try rephrasing your question."
            })
            return state
        
        def generate_answer(state: State) -> StateGraph:
            """
            Generate an answer based on the retrieved documents and user message.
            """
            context = state["context"]
            messages = state["messages"]
            
            
            # Combine context and messages to form the input for the LLM
            with open("./lecture-chatbot/chatbot/data/templates/generate_answer.txt", "r") as file:
                answer_template = file.read()

            input_text = answer_template.format(
                context="\n".join(doc["content"] for doc in context),
                messages="\n".join(f"{msg['role']}:{msg['content']}" for msg in messages)
            )

            result = self.llm.invoke(input_text).model_dump()
            result["role"] = "ai_assistant"
            state["messages"].append(result)

            return state
        
        
        def reformulate_query(state: State) -> State:
            user_question = state["messages"][-1]["content"]

            with open("./lecture-chatbot/chatbot/data/templates/reformulate_query.txt", "r") as file:
                reformulate_template = file.read()
            
            context = "\n".join(doc["content"] for doc in state["context"]) if hasattr(state, "context") else ""
            context = context.join(mess["content"] for mess in state["messages"][:-1])

            decision_prompt = reformulate_template.format(
                user_question=user_question, context=context
            )
            
            result = self.llm.invoke(decision_prompt).content.strip()

            if result != "NO_REWRITE":
                state["messages"][-1]["embedding_query"] = result
            else:
                state["messages"][-1]["embedding_query"] = user_question

            return state
        
        def should_retrieve_from_all_users(state: State) -> bool:
            """
            Determine if we should retrieve from all users based on the context.
            If the context is empty, we retrieve from all users.
            """
            return "retrieve_from_all_users" if len(state["context"]) == 0 else "generate_answer"
        
        def should_generate_answer(state: State) -> bool:
            """
            Determine if we should generate an answer based on the context.
            If the context is not empty, we generate an answer.
            """
            return "generate_answer" if len(state["context"]) > 0 else "no_document_found"

        graph.add_node("reformulate_query", reformulate_query)
        graph.add_node("retrieve", retrieve)
        graph.add_node("generate_answer", generate_answer)
        graph.add_node("retrieve_from_all_users", retrieve_from_all_users)
        graph.add_node("no_document_found", no_document_found)

        graph.add_edge(START, "reformulate_query")
        graph.add_edge("reformulate_query", "retrieve")

        # Go to generate_answer if context is not empty, otherwise retrieve from all users
        graph.add_conditional_edges("retrieve", 
            should_retrieve_from_all_users, 
            ["generate_answer", 
            "retrieve_from_all_users"]
        )

        graph.add_conditional_edges("retrieve_from_all_users", 
            should_generate_answer, 
            ["generate_answer", 
            "no_document_found"]
        )

        graph.add_edge("no_document_found", END)

        graph.add_edge("generate_answer", END)

        checkpointer = InMemorySaver()

        return graph.compile(checkpointer=checkpointer)
    
    def invoke(self, message: str, session_id, user: int) -> str:
        """
        Invoke the LLM with the given message and return the response.
        """

        #TODO: If there is already context from previous turns, do not necessarly call the rag and reuse context.

        config = {
            "configurable": {"thread_id": str(session_id)},
        }

        previous_state = self.graph.get_state(config).values
        if previous_state != {}:
            # Add the new user message to the previous state
            previous_state["messages"].append({"role": "user", "content": message})
            state = previous_state
        else:
            # Start a new conversation
            state = {
                "context": [],
                "messages": [{"role": "user", "content": message}],
                "user": user,
            }
    
        return self.graph.invoke(state, config=config)

