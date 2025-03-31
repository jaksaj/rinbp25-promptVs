# API Endpoint Examples

This document provides examples of all available API endpoints using curl commands.

## Base URL
All endpoints are prefixed with `http://localhost:8000`

## Models Endpoints

### List Available Models
```bash
curl http://localhost:8000/api/models
```

### List Running Models
```bash
curl http://localhost:8000/api/models/running
```

### Pull a Model
```bash
curl -X POST http://localhost:8000/api/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepseek-r1:1.5b"}'
```

### Start a Model
```bash
curl -X POST http://localhost:8000/api/models/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepseek-r1:1.5b"}'
```

### Stop a Model
```bash
curl -X POST http://localhost:8000/api/models/stop \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepseek-r1:1.5b"}'
```

## Prompts Endpoints

### Process a Prompt with a Specific Model
```bash
curl -X POST http://localhost:8000/api/prompt/deepseek-r1:1.5b \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

## Example Usage Flow

1. First, list available models:
```bash
curl http://localhost:8000/api/models
```

2. Pull a model if it's not available:
```bash
curl -X POST http://localhost:8000/api/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma3:1b"}'
```

3. Start the model:
```bash
curl -X POST http://localhost:8000/api/models/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma3:1b"}'
```

4. Check running models:
```bash
curl http://localhost:8000/api/models/running
```

5. Process a prompt with the running model:
```bash
curl -X POST http://localhost:8000/api/prompt/gemma3:1b \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

6. Stop the model when done:
```bash
curl -X POST http://localhost:8000/api/models/stop \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma3:1b"}'
```

## Notes
- The maximum number of running models is limited to 2
- Models must be pulled before they can be started
- Each prompt request must specify which model to use
- The server will return appropriate error messages if:
  - A model is not available
  - Too many models are running
  - A model is not running when trying to use it 