"""Export the API schema without starting services or model inference."""
import json
from _environment import ROOT
from assistant_app.server import app

if __name__ == "__main__":
    output = ROOT / "docs" / "openapi.json"
    output.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
