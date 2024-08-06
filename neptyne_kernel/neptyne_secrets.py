import warnings
from typing import Iterator


class EmptySecrets:
    def __getitem__(self, item: str) -> None:
        warnings.warn(
            "No current user. Cannot use user secrets in timed cell execution or tyne "
            "initialization. If you want to make a secret available for timed cells, use "
            "tyne-level secrets instead."
        )
        return None


class Secrets:
    secrets: dict[str, str]
    interactive: bool

    def __init__(self, secrets: dict, interactive: bool = False) -> None:
        self.secrets = secrets.copy()
        self.interactive = interactive

    def maybe_ask_for_secret(self, secret: str, help: str) -> str | None:
        from neptyne_kernel.kernel_runtime import get_kernel, in_gs_mode

        value = self.secrets.get(secret)
        if value:
            return value

        if not secret:
            raise KeyError("Empty key")

        if in_gs_mode():
            raise KeyError(
                f"Secret {secret} is not set. Choose Extensions -> Neptyne -> Manage Secrets to set it."
            )

        if not self.interactive:
            return EmptySecrets()[secret]

        kernel = get_kernel()
        ident = kernel._parent_ident["shell"]
        parent = kernel.get_parent("shell")
        parent["header"]["neptyne_secret_request"] = secret
        value = kernel._input_request(help, ident, parent, password=True)
        if not value:
            raise ValueError("Please enter a value for the secret")
        return value

    def __getitem__(self, item: str) -> str | None:
        return self.maybe_ask_for_secret(item, help="")

    def __iter__(self) -> Iterator[str]:
        return iter(self.secrets)
