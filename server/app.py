"""
Server entry point for OpenEnv validator compatibility.
Expects server/app.py with a main() function at project root.
"""
import uvicorn

from traffic_env.server.app import app  # noqa: F401


def main():
    """Start the environment server."""
    uvicorn.run(
        "traffic_env.server.app:app",
        host="0.0.0.0",
        port=7860,
    )


if __name__ == "__main__":
    main()
