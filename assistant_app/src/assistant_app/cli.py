"""Model-free argument parsing for the API server entrypoint."""
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the UNEC assistant API")
    parser.add_argument("--host", help="Override configured server host")
    parser.add_argument("--port", type=int, help="Override configured server port")
    args = parser.parse_args(argv)
    from .config import load_settings
    settings = load_settings()
    from .server import app
    import uvicorn
    uvicorn.run(app, host=args.host or settings.SERVER_HOST,
                port=args.port if args.port is not None else settings.SERVER_PORT,
                root_path=os.getenv("SERVER_ROOT_PATH", "").rstrip("/"))


if __name__ == "__main__":
    main()
