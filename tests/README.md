# Test Suite

This directory contains comprehensive tests for the Excel-Cursor MVP application.

## Test Structure

- `test_workbook.py` - Tests for workbook operations (create, read, update sheets)
- `test_formulas.py` - Tests for formula evaluation functionality
- `test_agent.py` - Tests for AI agent functionality
- `test_integration.py` - End-to-end integration tests
- `run_tests.py` - Test runner for executing all or specific tests

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

The test suite covers:
- ✅ Workbook creation and management
- ✅ Sheet operations (get, set, list, ensure)
- ✅ Cell operations (A1 notation)
- ✅ Formula evaluation (SUM, AVERAGE, COUNT, etc.)
- ✅ Mathematical operations
- ✅ String functions (CONCATENATE)
- ✅ Date functions (TODAY)
- ✅ AI agent functionality (mocked)
- ✅ Error handling
- ✅ End-to-end workflows

## Dependencies

Tests require:
- `unittest` (built-in)
- `pandas`
- `unittest.mock` (for AI agent tests)

## Adding New Tests

1. Create new test file following naming convention: `test_[feature].py`
2. Import required modules and create test class inheriting from `unittest.TestCase`
3. Add test methods with descriptive names starting with `test_`
4. Use `setUp()` method for test fixtures
5. Run tests to ensure they pass

## Continuous Integration

These tests can be integrated into CI/CD pipelines using:
```bash
python tests/run_tests.py
```

Exit code 0 indicates all tests passed, non-zero indicates failures.
