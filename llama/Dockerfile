FROM ollama/ollama:latest

WORKDIR /app
COPY setup.sh /app/setup.sh
RUN chmod +x /app/setup.sh && /app/setup.sh

CMD ["serve"]