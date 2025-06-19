#!/bin/bash

ollama serve &

until curl -s http://localhost:11434/ > /dev/null; do
    echo "Waiting for Ollama server..."
    sleep 1
done

ollama pull mistral:7b-instruct-q4_K_M
ollama pull nomic-embed-text:latest

wait -n