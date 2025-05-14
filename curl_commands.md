# cURL Commands for Testing Prompting Techniques

## Get Available Techniques

```bash
curl -X GET http://localhost:8000/api/techniques/techniques -H "Content-Type: application/json"
```

## Apply a Technique to a Prompt (Chain of Thought - Reasoning)

```bash
curl -X POST http://localhost:8000/api/techniques/apply \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_id": "your-existing-prompt-id",
    "technique": "cot_reasoning",
    "save_as_version": true
  }'
```

## Apply Few-Shot Technique

```bash
curl -X POST http://localhost:8000/api/techniques/apply \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_id": "your-existing-prompt-id",
    "technique": "few_shot",
    "save_as_version": true,
    "domain": "physics",
    "num_examples": 3
  }'
```

## Apply Self-Consistency Technique

```bash
curl -X POST http://localhost:8000/api/techniques/apply \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_id": "your-existing-prompt-id",
    "technique": "self_consistency",
    "save_as_version": true,
    "num_paths": 4
  }'
```

## View Technique Examples

```bash
curl -X GET http://localhost:8000/api/techniques/examples -H "Content-Type: application/json"
```

## Batch Apply Techniques

```bash
curl -X POST http://localhost:8000/api/techniques/batch-apply \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_ids": ["prompt-id-1", "prompt-id-2"],
    "techniques": ["cot_simple", "few_shot"],
    "save_as_versions": true,
    "domain": "computer-science",
    "num_examples": 3,
    "num_paths": 4
  }'
```

## PowerShell Versions

For PowerShell, use the following syntax:

### Get Available Techniques

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/techniques/techniques" -Method GET -ContentType "application/json"
```

### Apply a Technique to a Prompt

```powershell
$body = @{
    prompt_id = "your-existing-prompt-id"
    technique = "cot_reasoning"
    save_as_version = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/techniques/apply" -Method POST -Body $body -ContentType "application/json"
```

### Apply Few-Shot Technique

```powershell
$body = @{
    prompt_id = "your-existing-prompt-id"
    technique = "few_shot"
    save_as_version = $true
    domain = "physics"
    num_examples = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/techniques/apply" -Method POST -Body $body -ContentType "application/json"
```

### Apply Self-Consistency Technique

```powershell
$body = @{
    prompt_id = "your-existing-prompt-id"
    technique = "self_consistency"
    save_as_version = $true
    num_paths = 4
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/techniques/apply" -Method POST -Body $body -ContentType "application/json"
```

Replace `your-existing-prompt-id`, `prompt-id-1`, and `prompt-id-2` with your actual prompt IDs.
