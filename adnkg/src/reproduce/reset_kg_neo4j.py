import argparse
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase


def _get_env(key: str, fallbacks=None, required=False):
    fallbacks = fallbacks or []
    for env_key in [key, *fallbacks]:
        value = os.getenv(env_key)
        if value:
            return value
    if required:
        raise ValueError(f"Missing required environment variable: {key}")
    return None


def _confirm(force: bool):
    if force:
        return
    answer = input("Type RESET to confirm deleting all KG data: ")
    if answer.strip() != "RESET":
        print("Aborted.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Reset Neo4j KG data")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--database",
        type=str,
        default=os.getenv("NEO4J_DATABASE"),
        help="Neo4j database name (optional)",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=os.getenv("MINIRAG_ENV", ".env"), override=False)

    uri = _get_env("NEO4J_URI", required=True)
    user = _get_env("NEO4J_USERNAME", fallbacks=["NEO4J_USER"], required=True)
    password = _get_env("NEO4J_PASSWORD", required=True)
    database = args.database or None

    _confirm(args.force)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("KG reset complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
