from IPython import InteractiveShell


def get_ipython_mockable() -> InteractiveShell:
    # This exists so we can mock it in unit tests
    return get_ipython()  # type: ignore  # noqa: F821
