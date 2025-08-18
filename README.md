# Text Similarity Service

A scalable microservice for computing text similarity[^1] and LLM integration, built with FastAPI and Python.

## Features

- [x] **Text Similarity Metrics**[^1]: Cosine, Jaccard, and Semantic similarity
- [x] **Input / Output Sanitization**[^3]: Comprehensive safety measures for text processing
- [x] **LLM Integration**: Optional language model integration with retry logic
- [x] **Containerization**: Docker support with health checks
- [x] **Testing**: Unit tests, integration tests, and load testing

## Quick Start

### Prerequisites

- Python 3.13
- Docker (optional)
- Ollama (for LLM functionality)

### Local Development

#### 1. Clone the Repository

```bash
git clone https://github.com/jingkecn/Assessments.Python.TextSimilarity.git
cd text-similarity-service
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Start Ollama

```bash
ollama pull llama2
ollama serve
```

#### 4. Run the Service

```bash
python -m uvicorn app.main:app --reload
```

#### 5. Access the API

- API: <http://localhost:44101>
  - Metrics: <http://localhost:44101/metrics>
  - Similarity: <http://localhost:44101/similarity>
- Documentation (Swagger UI): <http://localhost:44101/docs>
- Health: <http://localhost:44101/health>

### Docker Deployment

Just run the following command under the repository directory:

```bash
./deploy.sh dev # or prod
```

## API Usage

### Calculate Text Similarity & LLM Integration

```http
POST /similarity HTTP/1.1
Host: localhost:44101
Content-Type: application/json

{
    "prompt1": "Who are you?",
    "prompt2": "Tell me about yourself.",
    "similarity_metric": "semantic",
    "similarity_threshold": 0.3,
    "use_llm": true
}
```

### Response Format (Example)

```json
{
    "are_similar": true,
    "llm_response": "I am LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner. I am trained on a massive dataset of text from the internet and can generate human-like responses to a wide range of topics and questions. I can be used to create chatbots, virtual assistants, and other applications that require natural language understanding and generation capabilities.",
    "similarity_metric": "semantic",
    "similarity_score": 0.38
}
```

## Testing

### Unit Tests

```bash
pytest tests/test_sanitization_service.py -v
pytest tests/test_similarity_service.py -v
```

### Integration Tests

> [!TODO] Test real integrated services instead of mocked ones.

```bash
pytest tests/test_api.py -v
```

### Load Testing

```bash
# Install Locust
pip install locust

# Run load test
locust -f scripts/load_test.py --host=http://localhost:44101
```

## Configuration

### Environment Variables

- For `dev` environment: [`.env.dev`](.env.dev)
- For `dev` environment: [`.env.prod`](.env.prod)

## Similarity Metrics

- [x] **Cosine Similarity**[^1]: Uses TF-IDF vectors, good for general text comparison
- [x] **Jaccard Similarity**[^1]: Based on word overlap, fast and simple
- [x] **Semantic Similarity**[^2]: Uses sentence transformers, best for meaning comparison

## Safety Features

- [x] **Input Sanitization**: Limits length
- [x] **Input / Output Sanitization**[^3]: Redacts forbidden phrases, harmful content, and sensitive information.

## Scaling Consideration

- [x] **Stateless Design**: Easy for horizontal scaling
- [x] **Async Processing**: Handles concurrent requests efficiently
- [ ] **Circuit Breakers**: Prevents cascade failures
- [x] **Health checks**: Kubernetes / Docker ready
- [ ] **Monitoring**: Structural logging for observability

## Performance

- [x] **Response Time**: ~10 ms (50th percentile)
- [ ] **LLM Calls**: TODO
- [x] **Throughput**: ~300 RPS (w/o LLM calls)
- [ ] **Memory**: TODO

[^1]: Aditya Singh (May 7, 2024). ["ULTIMATE GUIDE TO TEXT SIMILARITY WITH PYTHON"](https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python)
[^2]: Hugging Face. ["Sentence Similarity"](https://huggingface.co/tasks/sentence-similarity)
[^3]: Alex Drag, Head of Product Marketing, Kong (April 2, 2025). ["PII Sanitization Needed for LLMs and Agentic AI is Now Easier to Build"](https://konghq.com/blog/enterprise/building-pii-sanitization-for-llms-and-agentic-ai)
