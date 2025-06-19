# lecture-chatbot

This is a toy project to make use of LangChain/LangGraph and a Vector database. 

## Scope of the project

The objective for now will be to implement a RAG to answer users' questions on lectures.
The current behaviour that I will implement is the following : 
- User asks a question
- Look into database for lectures added by the user to answer the question
- If no result: look into the database for lectures from any user (or from default lectures)

The answer should precise the source, page, ..., and avoid answering when no relevant lecture is found to avoid hallucination.


## Technical specs

| Name | Spec |
|----------------|----------------|
|Vector database | ChromaDB |
|LLM used        | [mistral:7b-instruct-q4_K_M](https://ollama.com/library/mistral:7b-instruct-q4_K_M) |
|Embedding model | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) |