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

### Pull Multiple Models
```bash
curl -X POST http://localhost:8000/api/models/pull/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
  }'
```
Example response:
```json
[
    {
        "message": "Successfully pulled model: deepseek-r1:1.5b",
        "model_name": "deepseek-r1:1.5b"
    },
    {
        "message": "Successfully pulled model: gemma3:1b",
        "model_name": "gemma3:1b"
    },
    {
        "message": "Failed to pull model: llama3.2:1b",
        "model_name": "llama3.2:1b"
    }
]
```

### Start a Model
```bash
curl -X POST http://localhost:8000/api/models/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepseek-r1:1.5b"}'
```

### Start Multiple Models
```bash
curl -X POST http://localhost:8000/api/models/start/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["deepseek-r1:1.5b", "gemma3:1b"]
  }'
```
Example response:
```json
[
    {
        "message": "Successfully started model: deepseek-r1:1.5b",
        "model_name": "deepseek-r1:1.5b"
    },
    {
        "message": "Model gemma3:1b is not available",
        "model_name": "gemma3:1b"
    }
]
```

### Stop a Model
```bash
curl -X POST http://localhost:8000/api/models/stop \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepseek-r1:1.5b"}'
```

### Stop Multiple Models
```bash
curl -X POST http://localhost:8000/api/models/stop/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["deepseek-r1:1.5b", "gemma3:1b"]
  }'
```
Example response:
```json
[
    {
        "message": "Successfully stopped model: deepseek-r1:1.5b",
        "model_name": "deepseek-r1:1.5b"
    },
    {
        "message": "Failed to stop model: gemma3:1b",
        "model_name": "gemma3:1b"
    }
]
```

## Prompts Endpoints

### Process a Prompt with a Specific Model
```bash
curl -X POST http://localhost:8000/api/single/deepseek-r1:1.5b \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

### Process a Prompt with Multiple Models
```bash
curl -X POST http://localhost:8000/api/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "models": ["deepseek-r1:1.5b", "gemma3:1b"]
  }'
```
Example response:
```json
{
    "results": [
        {
            "model": "deepseek-r1:1.5b",
            "response": "The capital of France is Paris..."
        },
        {
            "model": "gemma3:1b",
            "response": "Error: Model gemma3:1b is not running"
        }
    ]
}
```

## Example Usage Flow

1. First, check the status of all models:
```bash
curl http://localhost:8000/api/models/status
```

2. Pull multiple models:
```bash
curl -X POST http://localhost:8000/api/models/pull/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["deepseek-r1:1.5b", "gemma3:1b"]
  }'
```

3. Check the status again to confirm the models are pulled:
```bash
curl http://localhost:8000/api/models/status
```

4. Start multiple models:
```bash
curl -X POST http://localhost:8000/api/models/start/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["deepseek-r1:1.5b", "gemma3:1b"]
  }'
```

5. Check running models:
```bash
curl http://localhost:8000/api/models/running
```

6. Process a prompt with multiple models:
```bash
curl -X POST http://localhost:8000/api/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "models": ["deepseek-r1:1.5b", "gemma3:1b"]
  }'
```

7. Stop multiple models when done:
```bash
curl -X POST http://localhost:8000/api/models/stop/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["deepseek-r1:1.5b", "gemma3:1b"]
  }'
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
- The batch prompt endpoint will process the prompt for all specified models that are running and return error messages for those that are not
- Batch operations (pull/start/stop) will process each model independently and return results for all models, even if some fail 