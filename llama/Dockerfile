FROM ollama/ollama:latest

WORKDIR /app
COPY setup.sh /app/setup.sh
RUN chmod +x /app/setup.sh && /app/setup.sh

CMD ["ollama", "serve", "--host", "0.0.0.0", "--port", "11434"]