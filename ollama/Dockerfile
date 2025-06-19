FROM ollama/ollama

RUN apt-get update && apt-get install -y build-essential
RUN apt-get install curl -y

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]