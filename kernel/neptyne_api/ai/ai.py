import json
import os
from functools import wraps
from typing import Callable

import requests

from ...kernel_runtime import get_api_token
from ...proxied_apis import TOKEN_HEADER_NAME
from ...renderers import create_with_source


def make_call(method, *args):
    host_port = os.getenv("API_PROXY_HOST_PORT", "localhost:8888")
    url = f"http://{host_port}/research/{method}"
    params = {"args": json.dumps(args)}
    headers = {TOKEN_HEADER_NAME: get_api_token()}

    payload = requests.get(url, params=params, headers=headers).json()
    if error := payload.get("error"):
        raise ValueError(error)
    value = payload["value"]
    if payload.get("source"):
        if isinstance(value, list):
            if isinstance(value[0], list):
                value = [
                    [
                        create_with_source(value[0][0], payload),
                        *value[0][1:],
                    ],
                    *value[1:],
                ]
            else:
                value = [
                    create_with_source(value[0], payload),
                    *value[1:],
                ]
        else:
            value = create_with_source(value, payload)
    return value


def _api_call(func: Callable):
    @wraps(func)
    def wrapper(*args):
        return make_call(func.__name__, *args)

    return wrapper


@_api_call
def value(query, *cells):
    """Ask the AI for a value. No source, so might not be accurate.

    query: the research topic
    cells: optional what the research refers to

    Example:
        =ai.value("The population of this city", A3)
        -- find the population of the city in A3. Returns a guess
           for smaller places or might just halucinate. Use
           ai.research for something backed by a source.
    """


@_api_call
def research(query, *cells):
    """Let the AI do research for you. Returns one value with a source.

    query: the research topic
    cells: optional what the research refers to

    Example:
        =ai.research("CEO of this company", A3)
        -- find the CEO of the company mentioned in A3
    """


@_api_call
def ai_list(query, count=None):
    """Returns a list of items matching the query, optionally limited to count.

    Example:
        =ai.list("countries in the EU", 27)
    """


@_api_call
def table(query: str, headers: list[str] | None = None, count: int | None = None):
    """Returns a table of items matching the query, optionally limited to count.

    Example:
        =ai.table("countries in the EU", ["name", "capital", "population", "gpd"], 27)
    """


def sources(*cells):
    """Formats a list of cells to use as source for a research call."""
    return ", ".join(cell.ref.range.origin().to_a1() for cell in cells)


__all__ = ["research", "value", "ai_list", "table", "sources"]
