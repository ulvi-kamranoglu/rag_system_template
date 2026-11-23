"""Server administration entrypoint using the active Python environment."""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from orchestration.cli import main

if __name__ == "__main__":
    main()
