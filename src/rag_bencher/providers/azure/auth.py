def is_installed() -> bool:
    try:
        import langchain_openai  # noqa: F401

        return True
    except Exception:
        return False
