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

## Prompt Versioning Endpoints (Neo4j)

### Create a New Prompt
```bash
curl -X POST http://localhost:8000/api/prompts \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Capital Cities",
    "description": "Prompts about capital cities",
    "tags": ["geography", "cities"]
  }'
```
Example response:
```json
{
    "id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Create a Prompt Version
```bash
curl -X POST http://localhost:8000/api/prompt-versions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "What is the capital of {country}?",
    "version": "1.0"
  }'
```
Example response:
```json
{
    "id": "660e8400-e29b-41d4-a716-446655440000"
}
```

### Create a Derived Prompt Version
```bash
curl -X POST http://localhost:8000/api/prompt-versions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "What is the capital city of {country} and what is its population?",
    "version": "1.1",
    "derived_from": "660e8400-e29b-41d4-a716-446655440000"
  }'
```

### Get Prompt Versions
```bash
curl http://localhost:8000/api/prompts/550e8400-e29b-41d4-a716-446655440000/versions
```
Example response:
```json
[
    {
        "id": "660e8400-e29b-41d4-a716-446655440000",
        "content": "What is the capital of {country}?",
        "version": "1.0",
        "created_at": "2023-11-01T12:34:56.789Z",
        "test_runs": 2
    },
    {
        "id": "770e8400-e29b-41d4-a716-446655440000",
        "content": "What is the capital city of {country} and what is its population?",
        "version": "1.1",
        "created_at": "2023-11-01T13:45:12.345Z",
        "test_runs": 1
    }
]
```

### Create a Test Run
```bash
curl -X POST http://localhost:8000/api/test-runs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_version_id": "660e8400-e29b-41d4-a716-446655440000",
    "model_used": "deepseek-r1:1.5b",
    "output": "The capital of France is Paris.",
    "metrics": {
      "latency_ms": 150,
      "token_count": 8,
      "token_per_second": 53.3,
      "custom_scores": {
        "relevance": 0.95,
        "accuracy": 1.0
      }
    },
    "input_params": {
      "temperature": 0.7,
      "country": "France"
    }
  }'
```

### Test a Prompt and Save Results
```bash
curl -X POST "http://localhost:8000/api/test-prompt-and-save?version_id=660e8400-e29b-41d4-a716-446655440000&model_name=deepseek-r1:1.5b&prompt=What%20is%20the%20capital%20of%20France?"
```

### Get Test Runs for a Prompt Version
```bash
curl http://localhost:8000/api/prompt-versions/660e8400-e29b-41d4-a716-446655440000/test-runs
```
Example response:
```json
[
    {
        "id": "880e8400-e29b-41d4-a716-446655440000",
        "model_used": "deepseek-r1:1.5b",
        "output": "The capital of France is Paris.",
        "metrics": {
            "latency_ms": 150,
            "token_count": 8,
            "token_per_second": 53.3,
            "custom_scores": {
                "relevance": 0.95,
                "accuracy": 1.0
            }
        },
        "created_at": "2023-11-01T14:22:33.456Z",
        "input_params": {
            "temperature": 0.7,
            "country": "France"
        }
    }
]
```

### Get a Prompt Version's Lineage
```bash
curl http://localhost:8000/api/prompt-versions/770e8400-e29b-41d4-a716-446655440000/lineage
```
Example response:
```json
[
    {
        "id": "660e8400-e29b-41d4-a716-446655440000",
        "content": "What is the capital of {country}?",
        "version": "1.0",
        "created_at": "2023-11-01T12:34:56.789Z",
        "depth": 1
    }
]
```

### Compare Test Runs
```bash
curl -X POST http://localhost:8000/api/compare-test-runs \
  -H "Content-Type: application/json" \
  -d '{
    "test_run_ids": [
      "880e8400-e29b-41d4-a716-446655440000",
      "990e8400-e29b-41d4-a716-446655440000"
    ]
  }'
```

### Search Test Runs
```bash
curl "http://localhost:8000/api/search-test-runs?query=Paris&model=deepseek-r1:1.5b"
```
Example response:
```json
[
    {
        "id": "880e8400-e29b-41d4-a716-446655440000",
        "model_used": "deepseek-r1:1.5b",
        "output": "The capital of France is Paris.",
        "metrics": {
            "latency_ms": 150,
            "token_count": 8,
            "token_per_second": 53.3,
            "custom_scores": {
                "relevance": 0.95,
                "accuracy": 1.0
            }
        },
        "created_at": "2023-11-01T14:22:33.456Z",
        "prompt_version": {
            "id": "660e8400-e29b-41d4-a716-446655440000",
            "content": "What is the capital of {country}?",
            "version": "1.0"
        },
        "input_params": {
            "temperature": 0.7,
            "country": "France"
        },
        "score": 0.87
    }
]
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

7. Create a prompt in Neo4j:
```bash
curl -X POST http://localhost:8000/api/prompts \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Geography Questions",
    "description": "Prompts about geography",
    "tags": ["geography", "education"]
  }'
```

8. Create a prompt version:
```bash
curl -X POST http://localhost:8000/api/prompt-versions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_id": "<PROMPT_ID>",
    "content": "What is the capital of {country}?",
    "version": "1.0"
  }'
```

9. Test the prompt and save the results:
```bash
curl -X POST "http://localhost:8000/api/test-prompt-and-save?version_id=<VERSION_ID>&model_name=deepseek-r1:1.5b&prompt=What%20is%20the%20capital%20of%20France?"
```

10. Get the test results:
```bash
curl http://localhost:8000/api/prompt-versions/<VERSION_ID>/test-runs
```

11. Stop multiple models when done:
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
- The Neo4j integration provides:
  - Prompt versioning and lineage tracking
  - Test run storage with metrics
  - Ability to compare different prompt versions
  - Full-text search across test run results