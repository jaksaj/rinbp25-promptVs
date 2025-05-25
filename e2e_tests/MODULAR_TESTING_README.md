# Modular Testing Scripts

This directory contains a modular approach to testing the PromptVs API, breaking down the resource-intensive E2E test into three smaller, focused scripts that can be run individually or in sequence.

## Overview

The modular testing system consists of three scripts that work together:

1. **01_create_prompts_and_versions.py** - Creates prompts and prompt versions
2. **02_create_test_runs.py** - Runs tests for each version
3. **03_ab_testing_evaluations.py** - Performs A/B testing evaluations and generates reports

Each script saves its results to JSON files that the next script can load, allowing for resource-efficient testing and the ability to restart from any point in the workflow.

## Scripts Description

### Script 1: Create Prompts and Versions
**File:** `01_create_prompts_and_versions.py`

**Purpose:** 
- Starts API server and Ollama
- Stops unnecessary models to conserve resources
- Creates prompt group and prompts
- Applies different techniques (cot_simple, cot_reasoning) to create versions

**Resource Usage:** Low (uses lightweight model for version creation)

**Output:** `prompts_and_versions.json`

**Usage:**
```bash
python 01_create_prompts_and_versions.py [options]
```

**Options:**
- `--api-url` - API base URL (default: http://localhost:8000)
- `--config` - Test configuration file (default: test_prompts_config.json)
- `--verbose` - Enable verbose logging

### Script 2: Create Test Runs
**File:** `02_create_test_runs.py`

**Purpose:**
- Loads prompts and versions from script 1
- Stops unnecessary models
- Starts test models (llama3.2:1b, gemma3:1b)
- Runs tests for each version multiple times

**Resource Usage:** Medium (uses test models for inference)

**Input:** `prompts_and_versions.json`
**Output:** `test_runs.json`

**Usage:**
```bash
python 02_create_test_runs.py [options]
```

**Options:**
- `--api-url` - API base URL (default: http://localhost:8000)
- `--input` - Input file from script 1 (default: prompts_and_versions.json)
- `--runs-per-version` - Number of test runs per version (default: 2)
- `--verbose` - Enable verbose logging

### Script 3: A/B Testing Evaluations
**File:** `03_ab_testing_evaluations.py`

**Purpose:**
- Loads test runs from script 2
- Stops unnecessary models
- Starts evaluation model (gemma3:4b)
- Performs A/B testing comparisons
- Generates comprehensive report

**Resource Usage:** High (uses larger model for evaluation quality)

**Input:** `test_runs.json`
**Output:** `evaluation_report_YYYYMMDD_HHMMSS.json`

**Usage:**
```bash
python 03_ab_testing_evaluations.py [options]
```

**Options:**
- `--api-url` - API base URL (default: http://localhost:8000)
- `--input` - Input file from script 2 (default: test_runs.json)
- `--verbose` - Enable verbose logging

## Convenience Scripts

### Run All Scripts in Sequence
**File:** `run_all_scripts.py`

Runs all three scripts in sequence with proper error handling and timing.

**Usage:**
```bash
python run_all_scripts.py [options]
```

**Options:**
- `--api-url` - API base URL (default: http://localhost:8000)
- `--config` - Test configuration file (default: test_prompts_config.json)
- `--runs-per-version` - Number of test runs per version (default: 2)
- `--verbose` - Enable verbose logging

### Windows Batch File
**File:** `run_all_scripts.bat`

Windows batch file for easy execution.

**Usage:**
```cmd
run_all_scripts.bat
```

## Resource Management

Each script is designed to be resource-conscious:

1. **Model Management:** Each script stops all running models before starting only the models it needs
2. **Memory Efficiency:** Data is saved to JSON files between scripts, allowing garbage collection
3. **Restart Capability:** You can restart from any script if a previous step completed successfully
4. **Model Selection:** Uses appropriately-sized models for each task (lightweight for generation, medium for testing, larger for evaluation)

## File Outputs

### prompts_and_versions.json
Contains:
- Prompt group ID
- List of prompt IDs
- Mapping of prompts to their versions
- Configuration data

### test_runs.json
Contains:
- All data from prompts_and_versions.json
- Mapping of versions to their test run IDs
- Test models used
- Total test run count

### evaluation_report_YYYYMMDD_HHMMSS.json
Contains:
- Complete test summary
- Technique performance analysis
- Best versions per prompt
- Detailed comparison results
- Win rates and statistics

## Example Workflow

### Run All Scripts at Once
```bash
# Run complete workflow with default settings
python run_all_scripts.py

# Run with custom settings
python run_all_scripts.py --runs-per-version 3 --verbose
```

### Run Scripts Individually
```bash
# Step 1: Create prompts and versions
python 01_create_prompts_and_versions.py --verbose

# Step 2: Create test runs (run 3 times per version)
python 02_create_test_runs.py --runs-per-version 3 --verbose

# Step 3: Perform evaluations
python 03_ab_testing_evaluations.py --verbose
```

### Restart from a Specific Point
```bash
# If script 1 completed but script 2 failed, restart from script 2
python 02_create_test_runs.py --input prompts_and_versions.json

# If scripts 1 and 2 completed but script 3 failed, restart from script 3
python 03_ab_testing_evaluations.py --input test_runs.json
```

## Advantages Over Monolithic E2E Test

1. **Resource Efficiency:** Each script uses only the resources it needs
2. **Fault Tolerance:** Can restart from any point without losing previous work
3. **Debugging:** Easier to debug issues in specific phases
4. **Flexibility:** Can adjust parameters for individual phases
5. **Monitoring:** Clear progress tracking and logging for each phase
6. **Scalability:** Can run different phases on different machines if needed

## Requirements

- Python 3.8+
- API server accessible (will auto-start if not running)
- Ollama installed and accessible
- Neo4j database running
- `test_prompts_config.json` configuration file

## Logging

Each script creates its own log file:
- `01_create_prompts_YYYYMMDD_HHMMSS.log`
- `02_test_runs_YYYYMMDD_HHMMSS.log`
- `03_evaluations_YYYYMMDD_HHMMSS.log`

## Error Handling

Each script includes comprehensive error handling:
- API connectivity issues
- Model availability problems
- Timeout handling for long-running operations
- Graceful degradation when individual operations fail
- Resource cleanup on exit

## Performance Estimates

Based on typical configurations:

- **Script 1:** 2-5 minutes (depends on technique complexity)
- **Script 2:** 10-30 minutes (depends on number of runs and model speed)
- **Script 3:** 5-20 minutes (depends on number of comparisons and evaluation model)

Total time is typically 20-60 minutes for a complete workflow, compared to the monolithic approach which could take 1-2 hours or more.
