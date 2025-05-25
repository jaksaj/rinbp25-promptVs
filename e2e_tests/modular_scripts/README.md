# Modular Testing Scripts

This directory contains the modular testing scripts that break down the resource-intensive E2E test into three focused components.

## Folder Structure

```
e2e_tests/
├── modular_scripts/        # The main testing scripts
│   ├── 01_create_prompts_and_versions.py
│   ├── 02_create_test_runs.py
│   ├── 03_ab_testing_evaluations.py
│   └── run_all_scripts.py
├── config/                 # Configuration files
│   └── test_prompts_config.json
├── output/                 # Generated output files
│   ├── prompts_and_versions.json
│   ├── test_runs.json
│   └── evaluation_report_*.json
├── logs/                   # Log files from script runs
│   ├── 01_create_prompts_*.log
│   ├── 02_test_runs_*.log
│   └── 03_evaluations_*.log
└── legacy/                 # Original E2E test files
    ├── e2e_test.py
    ├── debug_test.py
    ├── quick_test.py
    └── run_e2e_test.py
```

## Scripts Overview

### 1. Create Prompts and Versions
**Script:** `01_create_prompts_and_versions.py`

Creates prompt groups, prompts, and applies different techniques to generate versions.

**Usage:**
```bash
cd modular_scripts
python 01_create_prompts_and_versions.py [options]
```

**Options:**
- `--api-url` - API base URL (default: http://localhost:8000)
- `--config` - Configuration file path (default: ../config/test_prompts_config.json)
- `--verbose` - Enable verbose logging

**Output:** `../output/prompts_and_versions.json`

### 2. Create Test Runs
**Script:** `02_create_test_runs.py`

Runs tests for each prompt version using different models.

**Usage:**
```bash
cd modular_scripts
python 02_create_test_runs.py [options]
```

**Options:**
- `--api-url` - API base URL (default: http://localhost:8000)
- `--input` - Input file (default: ../output/prompts_and_versions.json)
- `--runs-per-version` - Number of test runs per version (default: 2)
- `--verbose` - Enable verbose logging

**Output:** `../output/test_runs.json`

### 3. A/B Testing Evaluations
**Script:** `03_ab_testing_evaluations.py`

Performs A/B testing comparisons and generates comprehensive reports.

**Usage:**
```bash
cd modular_scripts
python 03_ab_testing_evaluations.py [options]
```

**Options:**
- `--api-url` - API base URL (default: http://localhost:8000)
- `--input` - Input file (default: ../output/test_runs.json)
- `--verbose` - Enable verbose logging

**Output:** `../output/evaluation_report_YYYYMMDD_HHMMSS.json`

### 4. Run All Scripts
**Script:** `run_all_scripts.py`

Convenience script that runs all three scripts in sequence.

**Usage:**
```bash
cd modular_scripts
python run_all_scripts.py [options]
```

**Options:**
- `--api-url` - API base URL (default: http://localhost:8000)
- `--config` - Configuration file (default: ../config/test_prompts_config.json)
- `--runs-per-version` - Number of test runs per version (default: 2)
- `--verbose` - Enable verbose logging

## Quick Start

1. **Run all scripts in sequence:**
   ```bash
   cd modular_scripts
   python run_all_scripts.py --verbose
   ```

2. **Run scripts individually:**
   ```bash
   cd modular_scripts
   python 01_create_prompts_and_versions.py --verbose
   python 02_create_test_runs.py --runs-per-version 3 --verbose
   python 03_ab_testing_evaluations.py --verbose
   ```

3. **Restart from a specific point:**
   ```bash
   cd modular_scripts
   # If script 1 completed but script 2 failed
   python 02_create_test_runs.py --verbose
   
   # If scripts 1-2 completed but script 3 failed
   python 03_ab_testing_evaluations.py --verbose
   ```

## Resource Management

Each script automatically:
- Starts the API server if not running
- Starts Ollama service if not running
- Stops all models before starting to conserve resources
- Loads only the models it needs for its specific task
- Stops models after completion to free resources

## Model Usage by Script

- **Script 1:** `gemma3:1b` (lightweight model for prompt generation)
- **Script 2:** `llama3.2:1b`, `gemma3:1b` (test models)
- **Script 3:** `gemma3:4b` (evaluation model for quality comparisons)

## Output Files

### prompts_and_versions.json
Contains data from script 1:
- Prompt group ID
- List of prompt IDs
- Mapping of prompts to their versions
- Configuration data

### test_runs.json
Contains data from script 2:
- All data from prompts_and_versions.json
- Mapping of versions to test run IDs
- Test models used
- Total test run statistics

### evaluation_report_*.json
Contains comprehensive results from script 3:
- Test summary and statistics
- Technique performance analysis
- Best versions per prompt
- Detailed comparison results
- Win rates and performance metrics

## Log Files

Each script generates its own timestamped log file in the `../logs/` directory:
- `01_create_prompts_YYYYMMDD_HHMMSS.log`
- `02_test_runs_YYYYMMDD_HHMMSS.log`
- `03_evaluations_YYYYMMDD_HHMMSS.log`

## Error Handling

All scripts include:
- Comprehensive error handling for API and model issues
- Graceful degradation when individual operations fail
- Resource cleanup on exit
- Detailed error logging

## Performance

Typical execution times:
- **Script 1:** 2-5 minutes
- **Script 2:** 10-30 minutes (depends on runs-per-version)
- **Script 3:** 5-20 minutes (depends on number of comparisons)

Total workflow time: 20-60 minutes (vs 1-2+ hours for monolithic approach)

## Requirements

- Python 3.8+
- API server accessible
- Ollama installed
- Neo4j database running
- Configuration file in `../config/test_prompts_config.json`
