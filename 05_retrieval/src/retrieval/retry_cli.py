"""Publication retry used by the coordinator; never writes to Qdrant."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--publish-run", required=True)
    args = parser.parse_args()
    os.environ["RETRIEVAL_CONFIG"] = args.config
    from .storage import retry_publication
    retry_publication(args.publish_run, operation="sync")


if __name__ == "__main__":
    main()
