#!/usr/bin/env sh
set -e

echo "Starting temporary Ollama server to pull models..."

# Start damit die Modelle heruntergeladen werden können
ollama serve &

sleep 5

echo "Pulling models..."
ollama pull nomic-embed-text
ollama pull llama3

echo "Models pulled successfully."
