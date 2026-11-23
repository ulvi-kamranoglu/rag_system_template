# Assistant API

FastAPI, conversation routing and grounded Ollama generation. Retrieval is called
through the adapter; this package does not collect or modify datasets.

## Layout

- `src/assistant_app/`: Python implementation and packaged configuration defaults.
- `config/config.yaml`: active checkout configuration.
- `tests/`: model-free API, adapter, encoding and generation tests.
- `pyproject.toml`: package dependencies and command entrypoints.

## Commands

From the workspace root after `pip install -r requirements.txt`:

```powershell
assistant-server
python -m assistant_app.server
assistant-chat
python -m pytest assistant_app/tests -q
```

Server installation is documented in [deployment](../docs/deployment.md).
Endpoint contracts and frontend examples are in [the API guide](../docs/api.md).
