# lecture-chatbot

# Project informations
This is a toy project to make use of LangChain/LangGraph and a Vector database. 

## Scope of the project

The objective for now will be to implement a RAG to answer users' questions on lectures.
The current behaviour implemented is the following : 
- User asks a question
- Automatic rewrite of the question
- Look into database for lectures added by the user to answer the question
- If no result: look into the database for lectures from any user (or from default lectures)
- Answer question using found documents (or do not answer if no document is found)
- Handle multi-turn conversation

The answer should precise the source, page, ..., and avoid answering when no relevant lecture is found to avoid hallucination.

## Vector DB

This project uses ChromaDB with a persistent local client for simplicity. One can change this behaviour easily in the main.
Asynchronous calls of ChromaDB are not implemented as of now.

## Document ingestion

PDF import support

## Technical specs

| Name | Spec |
|----------------|----------------|
|Vector database | ChromaDB |
|LLM used        | [mistral:latest](https://ollama.com/library/mistral:latest) |
|Embedding model | [nomic-embed-text:latest](https://ollama.com/library/nomic-embed-text) |

## FastAPI commands

- Diverse debug commands for testing
- /file/upload_pdf to upload a PDF (mandatory before using the chatbot)
- /chat/infer to start talking to the chatbot

# Installation

## Main branch

### Ollama server
First install an Ollama server (example with WSL)
In WSL, run:
```
ollama serve
ollama pull mistral:latest
ollama pull nomic-embed-text:latest
```

Then, run following commands (from your chosen OS) -- suppose you are in /lecture-chatbot/ 

### Create env:
```
conda create -n lecture_chatbot python=3.12
conda activate lecture_chatbot
```

```pip install -r requirements.txt```

### Launch Application

```cd ..``` (if you were in /lecture-chatbot)
```uvicorn lecture-chatbot.main:app```

```ollama serve```

## Docker Branch

Go into /lecture-chatbot

Run:

```docker-compose up --build```

You are ready to use the application!

This will create 2 separate Docker images for Ollama and for the FastAPI application.

