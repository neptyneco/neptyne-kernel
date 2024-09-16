from ..dash import Dash


def connect_kernel(api_key: str, api_host="https://app.neptyne.com") -> None:
    """Connect the kernel to Neptyne. Used when running local kernels."""
    Dash.instance().initialize_local_kernel(api_key, api_host)
