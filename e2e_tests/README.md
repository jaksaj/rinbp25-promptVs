# E2E Test Suite

This folder contains the complete end-to-end test suite for the PromptVs API.

## Quick Start

### Windows (Recommended)
```cmd
run_tests.bat
```

### Python
```powershell
# Quick API verification
python quick_test.py

# Full end-to-end test
python e2e_test.py

# Or use the runner script
python run_e2e_test.py

#Script 1
python 01_create_prompts_and_versions.py --verbose

#Script 2
python 02_create_test_runs.py --runs-per-version 1 --verbose

#Script 3
python 03_ab_testing_evaluations.py --verbose
```

## Files

- **`e2e_test.py`** - Main E2E test script
- **`test_prompts_config.json`** - Test prompts configuration
- **`quick_test.py`** - Quick API verification
- **`run_e2e_test.py`** - Environment setup runner
- **`E2E_TEST_README.md`** - Detailed documentation
- **`TEST_SUITE_SUMMARY.md`** - Complete overview

## Prerequisites

1. **API Server**: Start from the main directory with `python run.py`
2. **Ollama**: Installed and accessible
3. **Neo4j**: Running database
4. **Python**: Required packages installed

## What the Test Does

1. ✅ Starts Ollama and required models
2. ✅ Creates a prompt group  
3. ✅ Creates 2 prompts from config file
4. ✅ Creates versions using `cot_simple` and `cot_reasoning`
5. ✅ Runs each version 2 times with 2 models
6. ✅ Evaluates all test runs
7. ✅ Compares versions and finds best performers
8. ✅ Generates comprehensive report

## Output

- **Log file**: `e2e_test_YYYYMMDD_HHMMSS.log`
- **Report file**: `e2e_test_report_YYYYMMDD_HHMMSS.json`

See `E2E_TEST_README.md` for complete documentation.
