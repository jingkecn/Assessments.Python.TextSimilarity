#!/bin/bash

# Check if environment argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <environment>"
    echo "Available environments: dev, prod"
    exit 1
fi

ENV_TO_DEPLOY=$1

# Validate environment
if [ "$ENV_TO_DEPLOY" != "dev" ] && [ "$ENV_TO_DEPLOY" != "prod" ]; then
    echo "Error: Invalid environment '$ENV_TO_DEPLOY'"
    echo "Available environments: dev, prod"
    exit 1
fi

# Check if env file exists
if [ ! -f ".env.$ENV_TO_DEPLOY" ]; then
    echo "Error: .env.$ENV_TO_DEPLOY file not found"
    exit 1
fi

# Load environment variables from .env file
set -o allexport
source .env."$ENV_TO_DEPLOY"
set +o allexport

echo "Setting up environment: $ENV_TO_DEPLOY"

# Tear down existing containers
docker compose -f docker-compose.yml -f docker-compose."$ENV_TO_DEPLOY".yml down

# Build and start Docker containers
docker compose -f docker-compose.yml -f docker-compose."$ENV_TO_DEPLOY".yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose."$ENV_TO_DEPLOY".yml up -d

# Download the Ollama model
echo "Downloading Ollama model: $LLM_MODEL"
docker compose exec ollama ollama pull "$LLM_MODEL"

# Verify the model is downloaded
docker compose exec ollama ollama list
