import os
import urllib.parse
from collections import defaultdict
from typing import Any, Callable

import httpx
import requests

from .kernel_runtime import get_api_token, send_out_of_quota

API_EXCEEDED_MESSAGE_PREFIX = "Neptyne API limit exceeded for service=["
PLACEHOLDER_API_KEY = "NEPTYNE_PLACEHOLDER"


TOKEN_HEADER_NAME = "X-Neptyne-Token"

NEEDS_GSHEET_ADVANCED_FEATURES_HTTP_CODE = 499


def get_api_error_service(error: str) -> str | None:
    if (ix := error.find(API_EXCEEDED_MESSAGE_PREFIX)) != -1:
        service = error[ix + len(API_EXCEEDED_MESSAGE_PREFIX) :]
        return service[: service.find("]")]


def make_api_error_message(service: str) -> str:
    return f"{API_EXCEEDED_MESSAGE_PREFIX}{service}]"


def start_api_proxying() -> None:
    proxied_urls: defaultdict[str, list] = defaultdict(list)
    host_port = os.getenv("API_PROXY_HOST_PORT", "localhost:8888")

    if os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = PLACEHOLDER_API_KEY
        # os.environ["OPENAI_BASE_URL"] = f"http://{host_port}/openai/v1"

    def new_send_method(
        original_send_method: Callable,
        self: requests.sessions.Session | httpx.Client,
        request: requests.Request | httpx.Request,
        *args: Any,
        **kwargs: Any,
    ) -> requests.Response | httpx.Response:
        if isinstance(request.url, str):
            _scheme, netloc, path, *rest = urllib.parse.urlparse(request.url)
        elif isinstance(request.url, httpx.URL):
            netloc = request.url.host
            path = request.url.path
            rest = []
        else:
            raise ValueError(f"Unexpected request.url type: {type(request.url)}")
        if request.headers is None:
            request.headers = {}

        if "timeout" in kwargs and kwargs["timeout"] is not None:
            request.headers["X-Neptyne-Timeout"] = str(kwargs["timeout"])

        patched = False
        for service_name, proxied_path in proxied_urls.get(netloc.lower(), []):
            if path.startswith(proxied_path):
                if api_token := get_api_token():
                    request.headers[TOKEN_HEADER_NAME] = api_token
                if isinstance(request.url, str):
                    request.url = urllib.parse.urlunparse(
                        ("http", host_port, "/" + service_name + path, *rest)
                    )
                else:
                    port: str | int | None = None
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                        port = int(port)
                    else:
                        host = host_port
                    request.url = httpx.URL(
                        request.url,  # type: ignore
                        scheme="http",
                        host=host,
                        port=port,
                        path="/" + service_name + path,
                    )
                patched = True
                break
        response = original_send_method(self, request, *args, **kwargs)
        if (
            patched
            and response.status_code == 429
            and response.headers.get("X-Neptyne-Error") == "Quota exceeded"
        ):
            send_out_of_quota()
        return response

    def patch_cls(cls: type[requests.sessions.Session | httpx.Client]) -> None:
        org_method = cls.send

        def do_call(
            self: requests.sessions.Session | httpx.Client,
            request: httpx.Request | requests.Request,
            *args: Any,
            **kwargs: Any,
        ) -> requests.Response | httpx.Response:
            return new_send_method(org_method, self, request, *args, **kwargs)

        cls.send = do_call  # type: ignore

    patch_cls(requests.sessions.Session)
    patch_cls(httpx.Client)

    def proxy_url(
        service_name_to_proxy: str,
        url: str,
        send_out_of_quota: Callable[[], None],
        env_vars_to_set: dict[str, str] | None = None,
    ) -> None:
        parsed_url = urllib.parse.urlparse(url)
        proxied_urls[parsed_url.netloc.lower()].append(
            (service_name_to_proxy, parsed_url.path)
        )
        if env_vars_to_set:
            for k, v in env_vars_to_set.items():
                os.environ[k] = v

    proxy_url(
        "openai",
        "https://api.openai.com/v1/",
        send_out_of_quota,
        {"OPENAI_API_KEY": PLACEHOLDER_API_KEY},
    )
    proxy_url(
        "anthropic",
        "https://api.anthropic.com/v1/",
        send_out_of_quota,
        {"ANTHROPIC_API_KEY": PLACEHOLDER_API_KEY},
    )
    proxy_url(
        "phantomjscloud",
        "https://PhantomJsCloud.com/api/",
        send_out_of_quota,
    )
    proxy_url(
        "bing",
        "https://api.bing.microsoft.com/v7.0/search",
        send_out_of_quota,
    )
    proxy_url(
        "google_maps_geocode",
        "https://maps.googleapis.com/maps/api/geocode/json",
        send_out_of_quota,
    )
    proxy_url(
        "iexfinance",
        "https://cloud.iexapis.com/",
        send_out_of_quota,
        {"IEX_TOKEN": PLACEHOLDER_API_KEY},
    )
    proxy_url(
        "google_ai",
        "https://generativelanguage.googleapis.com/",
        send_out_of_quota,
    )
    try:
        import google.generativeai as genai

        genai.configure(
            api_key=PLACEHOLDER_API_KEY,
            transport="rest",
        )
    except ImportError:
        pass
