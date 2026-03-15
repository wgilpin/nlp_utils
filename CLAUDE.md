# Project Instructions for Claude

## Package Management
- Always use `uv` for everything rather than `pip`
  - Use `uv add` instead of `pip install`
  - Use `uv run` to execute Python scripts
  - Use `uv sync` to install dependencies

## Testing
- Add tests for all library public functions
- Use TDD - create failing tests first, then write the function to pass the test
- Mock any LLM calls or URL fetches in tests
  - Use appropriate mocking libraries (e.g., `unittest.mock`, `pytest-mock`)
  - Do not make actual API calls or network requests in unit tests
