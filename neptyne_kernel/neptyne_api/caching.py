from typing import Callable


class _CachingDecorator:
    def __init__(self, caching_type):
        self.caching_type = caching_type

    def __call__(self, func):
        func.caching = self.caching_type
        return func


class cache:
    """Decorator to specify the caching type of a function. Don't use directly, us one of the members instead.

    Example:
    ```
    @cache.never
    def my_function():
        pass
    ```

    Valid caching types are "never" and "always".

    """

    def __call__(self, *args, **kwargs):
        raise ValueError(
            "Specify the caching type: never or always, like @nt.cache.never"
        )

    @classmethod
    def _set(cls, fn: Callable, caching_type: str):
        fn.caching = caching_type

    @classmethod
    def never(cls, fn: Callable) -> Callable:
        """Never cache the function: @nt.cache.never"""
        cls._set(fn, "never")
        return fn

    @classmethod
    def always(cls, fn: Callable) -> Callable:
        """Always cache the function: @nt.cache.always"""
        cls._set(fn, "always")
        return fn
