# Test Suite

This directory contains comprehensive tests for the Excel-Cursor MVP application with AI-powered data generation.

## Test Structure

### Core Functionality Tests
- `test_workbook.py` - Tests for workbook operations (create, read, update sheets)
- `test_formulas.py` - Tests for formula evaluation functionality (SUM, AVERAGE, COUNT, etc.)
- `test_integration.py` - End-to-end integration tests and workflows

### AI Agent Tests
- `test_agent.py` - Tests for AI agent functionality, error handling, and model probing
- `test_agent_tools.py` - Comprehensive tests for all 11 AI tools including data generation
- `test_function_calling.py` - Tests for OpenAI function calling mechanisms
- `test_tool_calling.py` - Tool calling validation and error handling

### Data Generation Tests
- `test_data_generation.py` - Data generation testing utilities and validation
- `test_product_generation.py` - Specific product data generation testing
- `test_final.py` - Final integration validation for complete system

### Utilities
- `run_tests.py` - Test runner for executing all or specific tests
- `debug_agent_tools.py` - Debug utilities for agent development

## Running Tests

### Run all tests:
```bash
cd tests
python run_tests.py
```

### Run specific test module:
```bash
cd tests
python run_tests.py test_workbook
```

### Run individual test files:
```bash
cd tests
python -m unittest test_workbook.py
python -m unittest test_formulas.py
python -m unittest test_agent.py
python -m unittest test_integration.py
```

## Test Coverage

The comprehensive test suite covers:

### Core Functionality (36+ Tests)
- ✅ **Workbook Management**: Creation, sheet operations, data manipulation
- ✅ **Formula Engine**: SUM, AVERAGE, COUNT, MIN, MAX, TODAY, CONCATENATE
- ✅ **Cell Operations**: A1 notation, get/set values, data types
- ✅ **Mathematical Operations**: Basic arithmetic, complex expressions
- ✅ **Error Handling**: Invalid formulas, missing sheets, cell references

### AI-Powered Features (19+ Tests)
- ✅ **Agent Functionality**: GPT-4 integration, timeout handling, model probing
- ✅ **Tool Execution**: All 11 AI tools (set_cell, generate_sample_data, etc.)
- ✅ **Data Generation**: Context-aware sample data for restaurants, employees, products
- ✅ **Function Calling**: OpenAI function calling mechanisms and error recovery
- ✅ **Intelligent Templates**: Fallback system for when AI is unavailable

### Integration & Workflows (8+ Tests)
- ✅ **End-to-End**: Complete data manipulation workflows
- ✅ **Agent Tools**: CSV creation, sheet management, chart generation
- ✅ **Data Quality**: Professional formatting, realistic business data
- ✅ **Error Recovery**: Graceful handling of AI failures and invalid operations

### Quality Assurance
- ✅ **Enterprise Reliability**: Production-ready error handling
- ✅ **Performance**: Efficient data processing and memory management
- ✅ **Compatibility**: Works with multiple OpenAI models (GPT-4, o1, etc.)
- ✅ **Comprehensive Coverage**: Tests both happy path and edge cases

## Dependencies

Tests require:
- `unittest` (built-in)
- `pandas` (data manipulation)
- `openai` (AI agent functionality)
- `python-dotenv` (environment variables)
- `unittest.mock` (for mocking external services)

## Environment Setup

For full AI functionality testing:
```bash
# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"

# Or create .env file in project root
echo "OPENAI_API_KEY=your-api-key-here" > ../.env
```

## Test Categories

### Quick Tests (No API calls)
- Core workbook and formula tests
- Template-based data generation
- Error handling validation

### Full AI Tests (Requires API key)
- OpenAI GPT-4 integration
- AI-powered data generation
- Model probing and selection

## Adding New Tests

1. Create new test file following naming convention: `test_[feature].py`
2. Import required modules and create test class inheriting from `unittest.TestCase`
3. Add test methods with descriptive names starting with `test_`
4. Use `setUp()` method for test fixtures
5. Run tests to ensure they pass

## Continuous Integration

These tests can be integrated into CI/CD pipelines using:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python -m unittest tests.test_workbook
python -m unittest tests.test_formulas
python -m unittest tests.test_agent_tools
```

**Exit Codes:**
- `0` - All tests passed
- `1` - Some tests failed

## Test Results

Current test status: **36/36 tests passing** ✅

The test suite validates:
- Core Excel-like functionality
- AI-powered data generation
- Enterprise-grade error handling
- Production-ready reliability

For detailed test output, run with verbose mode to see individual test results and performance metrics.
