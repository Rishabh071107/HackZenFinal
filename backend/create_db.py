import sys
from pathlib import Path

# Add backend directory to path
backend_path = Path(__file__).resolve().parent
sys.path.append(str(backend_path))

import psycopg2  # noqa: E402
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT  # noqa: E402

from app.core.config import get_settings  # noqa: E402


def main():
    settings = get_settings()
    dbname = settings.POSTGRES_DB
    user = settings.POSTGRES_USER
    password = settings.POSTGRES_PASSWORD
    host = settings.POSTGRES_HOST
    port = settings.POSTGRES_PORT

    print(f"Connecting to database server on {host}:{port} as {user}...")
    try:
        # Connect to the default 'postgres' database first to check/create the target database
        conn = psycopg2.connect(
            dbname="postgres", user=user, password=password, host=host, port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s;", (dbname,)
        )
        exists = cursor.fetchone()

        if not exists:
            print(f"Database '{dbname}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {dbname};")
            print(f"Database '{dbname}' created successfully.")
        else:
            print(f"Database '{dbname}' already exists.")

        cursor.close()
        conn.close()
    except Exception as e:
        print(
            f"\nError connecting to PostgreSQL/creating database: {e}", file=sys.stderr
        )
        print("\nPlease verify that:", file=sys.stderr)
        print("1. The local PostgreSQL service is running.", file=sys.stderr)
        print("2. The credentials/port in backend/.env are correct.", file=sys.stderr)
        print(f"Current config: Host={host}, Port={port}, User={user}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
