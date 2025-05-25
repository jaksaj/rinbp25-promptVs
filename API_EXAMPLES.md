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
    "name": "llama3.2:1b",
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
  -d '{"model_name": "llama3.2:1b"}'
```

### Pull Multiple Models
```bash
curl -X POST http://localhost:8000/api/models/pull/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["llama3.2:1b", "gemma3:1b", "llama3.2:1b"]
  }'
```
Example response:
```json
[
    {
        "message": "Successfully pulled model: llama3.2:1b",
        "model_name": "llama3.2:1b"
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
  -d '{"model_name": "llama3.2:1b"}'
```

### Start Multiple Models
```bash
curl -X POST http://localhost:8000/api/models/start/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["llama3.2:1b", "gemma3:1b"]
  }'
```
Example response:
```json
[
    {
        "message": "Successfully started model: llama3.2:1b",
        "model_name": "llama3.2:1b"
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
  -d '{"model_name": "llama3.2:1b"}'
```

### Stop Multiple Models
```bash
curl -X POST http://localhost:8000/api/models/stop/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["llama3.2:1b", "gemma3:1b"]
  }'
```
Example response:
```json
[
    {
        "message": "Successfully stopped model: llama3.2:1b",
        "model_name": "llama3.2:1b"
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
curl -X POST http://localhost:8000/api/single/llama3.2:1b \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

### Process a Prompt with Multiple Models
```bash
curl -X POST http://localhost:8000/api/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "models": ["llama3.2:1b", "gemma3:1b"]
  }'
```
Example response:
```json
{
    "results": [
        {
            "model": "llama3.2:1b",
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
    "model_used": "llama3.2:1b",
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
curl -X POST "http://localhost:8000/api/test-prompt-and-save?version_id=660e8400-e29b-41d4-a716-446655440000&model_name=llama3.2:1b&prompt=What%20is%20the%20capital%20of%20France?"
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
        "model_used": "llama3.2:1b",
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
curl "http://localhost:8000/api/search-test-runs?query=Paris&model=llama3.2:1b"
```
Example response:
```json
[
    {
        "id": "880e8400-e29b-41d4-a716-446655440000",
        "model_used": "llama3.2:1b",
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

## Prompt Generation Endpoints

### Generate Prompt Variations
```bash
curl -X POST http://localhost:8000/api/generation/generate-variations \
  -H "Content-Type: application/json" \
  -d '{
    "base_prompt": "What is the capital of {country}?",
    "num_variations": 3,
    "variation_type": "rephrase",
    "template_params": {"country": "placeholder"}
  }'
```
Example response:
```json
{
  "variations": [
    "Could you tell me what the capital of {country} is?",
    "What city serves as the capital of {country}?",
    "I need to know the capital city of {country}, what is it?"
  ],
  "original_prompt": "What is the capital of {country}?",
  "variation_type": "rephrase",
  "generation_time": 1.245
}
```

### Generate Multiple Prompt Variations in Batch
```bash
curl -X POST http://localhost:8000/api/generation/batch-generate-variations \
  -H "Content-Type: application/json" \
  -d '{
    "base_prompts": [
      "What is the capital of {country}?",
      "Who is the president of {country}?"
    ],
    "num_variations_each": 2,
    "variation_types": ["formalize", "casual"],
    "template_params": {"country": "placeholder"},
    "generator_model": "llama3:8b"
  }'
```
Example response:
```json
{
  "results": [
    {
      "variations": [
        "I would like to inquire about the capital city of {country}. Could you please provide me with this information?",
        "May I request information regarding the capital of {country}?"
      ],
      "original_prompt": "What is the capital of {country}?",
      "variation_type": "formalize",
      "generation_time": 1.834
    },
    {
      "variations": [
        "Hey, what's the capital of {country}?",
        "So, tell me, what city is the capital of {country}?"
      ],
      "original_prompt": "What is the capital of {country}?",
      "variation_type": "casual",
      "generation_time": 1.523
    },
    {
      "variations": [
        "I would like to inquire about the current head of state of {country}. Could you please inform me who holds the office of president?",
        "Could you kindly provide information regarding the individual who currently serves as the president of {country}?"
      ],
      "original_prompt": "Who is the president of {country}?",
      "variation_type": "formalize",
      "generation_time": 1.945
    },
    {
      "variations": [
        "Hey, who's running {country} these days? Like, who's the president?",
        "So who's the big boss of {country} right now? The president, I mean."
      ],
      "original_prompt": "Who is the president of {country}?",
      "variation_type": "casual",
      "generation_time": 1.632
    }
  ],
  "total_variations": 8,
  "failed_prompts": []
}
```

### Analyze a Prompt
```bash
curl -X POST http://localhost:8000/api/generation/analyze-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of {country}?",
    "generator_model": "llama3:8b"
  }'
```
Example response:
```json
{
  "clarity": 9,
  "specificity": 8,
  "potential_issues": [
    "The prompt doesn't specify what level of detail is expected in the answer",
    "Some countries might have disputed capitals or multiple capital cities"
  ],
  "improvement_suggestions": [
    "Consider specifying whether you want just the name or additional information",
    "You could clarify whether you want the administrative capital, legislative capital, or both for countries with multiple capitals"
  ],
  "template_parameters": [
    "country"
  ]
}
```

### Generate Variations and Create Versions
```bash
curl -X POST "http://localhost:8000/api/generation/generate-and-create-versions?prompt_id=550e8400-e29b-41d4-a716-446655440000" \
  -H "Content-Type: application/json" \
  -d '{
    "base_prompt": "What is the capital of {country}?",
    "num_variations": 3,
    "variation_type": "multiple_choice",
    "template_params": {"country": "placeholder"}
  }'
```
Example response:
```json
{
  "created_versions": [
    {
      "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
      "version_id": "770e8400-e29b-41d4-a716-446655440001",
      "version": "multiple_choice_1",
      "content": "Which city is the capital of {country}?\nA) Paris\nB) London\nC) Berlin\nD) Rome\nE) None of the above"
    },
    {
      "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
      "version_id": "770e8400-e29b-41d4-a716-446655440002",
      "version": "multiple_choice_2",
      "content": "From the following options, select the capital city of {country}:\n1. Tokyo\n2. Beijing\n3. Moscow\n4. Madrid\n5. Other"
    },
    {
      "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
      "version_id": "770e8400-e29b-41d4-a716-446655440003",
      "version": "multiple_choice_3",
      "content": "The capital of {country} is:\na) New York\nb) Cairo\nc) Sydney\nd) Bangkok\ne) None of these"
    }
  ],
  "failed_versions": [],
  "total_created": 3,
  "total_failed": 0,
  "generation_time": 2.347,
  "variation_type": "multiple_choice"
}
```

## Extended Usage Flow with Prompt Generation

Building on the previous example flow, here's how to incorporate prompt generation:

12. After creating a basic prompt version, generate variations of it:
```bash
curl -X POST http://localhost:8000/api/generation/generate-variations \
  -H "Content-Type: application/json" \
  -d '{
    "base_prompt": "What is the capital of {country}?",
    "num_variations": 3,
    "variation_type": "add_context",
    "template_params": {"country": "placeholder"}
  }'
```

13. Select the variation you prefer and create a new version from it:
```bash
curl -X POST http://localhost:8000/api/prompt-versions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_id": "<PROMPT_ID>",
    "content": "Given that countries can change their capitals over time, what is the current official capital city of {country}?",
    "version": "1.1",
    "derived_from": "<ORIGINAL_VERSION_ID>"
  }'
```

14. Or use the combined endpoint to generate variations and create versions all at once:
```bash
curl -X POST "http://localhost:8000/api/generation/generate-and-create-versions?prompt_id=<PROMPT_ID>" \
  -H "Content-Type: application/json" \
  -d '{
    "base_prompt": "What is the capital of {country}?",
    "num_variations": 3,
    "variation_type": "add_irrelevant",
    "template_params": {"country": "placeholder"}
  }'
```

15. Run tests on all versions to compare performance:
```bash
curl -X POST http://localhost:8000/api/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_version_ids": ["<VERSION_ID_1>", "<VERSION_ID_2>", "<VERSION_ID_3>"],
    "models": ["llama3.2:1b"]
  }'
```