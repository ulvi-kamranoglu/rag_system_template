import json
from assistant_app.server import JSONResponse


def test_json_declares_utf8_and_preserves_azerbaijani():
    text = "UNEC-in rektoru professor Ədalət Muradovdur."
    response = JSONResponse({"answer": text})
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert json.loads(response.body.decode("utf-8"))["answer"] == text
