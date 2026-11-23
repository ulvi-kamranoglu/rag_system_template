"""Read-only artifact/service checks. No model loading or index mutations."""
import argparse
import json

from _environment import ROOT, snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Check local files/configuration only")
    args = parser.parse_args(argv)
    from assistant_app.config import load_settings as assistant_settings
    from retrieval.config import load_settings
    settings, assistant = load_settings(), assistant_settings()
    checks = []

    def check(name, operation):
        try:
            detail = operation()
            checks.append({"check": name, "ok": True, "detail": detail})
        except Exception as exc:
            checks.append({"check": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})

    desired = None

    def local():
        nonlocal desired
        count, desired = snapshot(settings)
        return {"dataset": settings["DATASET_NAME"], "chunks": count,
                "dimension": settings["VECTOR_DIMENSION"]}

    check("local_snapshot", local)

    if not args.offline:
        def qdrant():
            from qdrant_client import QdrantClient
            client = QdrantClient(url=settings["QDRANT_URL"], timeout=30)
            try:
                collection = settings["QDRANT_COLLECTION"]
                info = client.get_collection(collection)
                params = info.config.params.vectors
                distance = str(getattr(getattr(params, "distance", ""), "value", getattr(params, "distance", ""))).lower()
                if getattr(params, "size", None) != settings["VECTOR_DIMENSION"] or distance != "cosine":
                    raise ValueError("Qdrant vector configuration differs")
                actual, offset = set(), None
                point_count = 0
                while True:
                    points, offset = client.scroll(collection, offset=offset, limit=512,
                                                   with_payload=["chunk_id"], with_vectors=False)
                    for point in points:
                        point_count += 1
                        chunk_id = (point.payload or {}).get("chunk_id")
                        if not isinstance(chunk_id, str) or not chunk_id:
                            raise ValueError("Qdrant point missing chunk_id")
                        actual.add(chunk_id)
                    if offset is None:
                        break
                count = client.count(collection, exact=True).count
                if desired is None:
                    raise ValueError("Local snapshot failed; cannot verify remote identity")
                if actual != desired or count != len(desired) or point_count != len(desired):
                    raise ValueError("Qdrant and local snapshot chunk IDs/count differ")
                return {"collection": collection, "points": count}
            finally:
                client.close()

        def ollama():
            import requests
            response = requests.get(assistant.OLLAMA_URL + "/api/tags", timeout=15)
            response.raise_for_status()
            names = {item.get("name") for item in response.json().get("models", [])}
            expected = assistant.LLM_MODEL
            if expected not in names and expected + ":latest" not in names:
                raise ValueError(f"Configured model is not installed: {expected}")
            return {"model": expected, "installed": True}

        check("qdrant_snapshot", qdrant)
        check("ollama_model", ollama)

    print(json.dumps({"project_root": str(ROOT), "offline": args.offline, "checks": checks},
                     ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
