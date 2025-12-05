def is_installed() -> bool:
    try:
        import langchain_google_vertexai  # noqa: F401

        return True
    except Exception:
        return False
