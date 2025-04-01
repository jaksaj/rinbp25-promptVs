# API Endpoint Examples

This document provides examples of all available API endpoints using curl commands.

## Base URL
All endpoints are prefixed with `http://localhost:8000`

## Models Endpoints

### List Available Models
```bash
curl http://localhost:8000/api/models
```

### Get Models Status
```bash
curl http://localhost:8000/api/models/status
```
Example response:
```json
[
  {
    "name": "deepseek-r1:1.5b",
    "is_pulled": true,
    "is_running": true,
    "status_message": "Running"
  },
  {
    "name": "gemma3:1b",
    "is_pulled": false,
    "is_running": false,
    "status_message": "Not pulled"
  },
  {
    "name": "llama3.2:1b",
    "is_pulled": true,
    "is_running": false,
    "status_message": "Pulled but not running"
  }
]
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

1. First, check the status of all models:
```bash
curl http://localhost:8000/api/models/status
```

2. Pull a model if it's not available:
```bash
curl -X POST http://localhost:8000/api/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma3:1b"}'
```

3. Check the status again to confirm the model is pulled:
```bash
curl http://localhost:8000/api/models/status
```

4. Start the model:
```bash
curl -X POST http://localhost:8000/api/models/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma3:1b"}'
```

5. Check running models:
```bash
curl http://localhost:8000/api/models/running
```

6. Process a prompt with the running model:
```bash
curl -X POST http://localhost:8000/api/prompt/gemma3:1b \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

7. Stop the model when done:
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
  - A model is already being pulled/started/stopped
- The `/models/status` endpoint provides detailed information about each model's state 