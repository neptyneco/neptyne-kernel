import os
import subprocess
import sys
from tempfile import NamedTemporaryFile


def neptyne_pip_install(requirements_txt: str, silent: bool = False) -> None:
    f = NamedTemporaryFile(delete=False)
    try:
        f.write(requirements_txt.encode())
        f.close()
        res = subprocess.run(
            [
                "pip",
                "--disable-pip-version-check",
                "--no-color",
                "install",
                "--no-warn-script-location",
                "--progress-bar=off",
                "--root-user-action=ignore",
                "-r",
                f.name,
            ],
            capture_output=silent,
        )
        if res.returncode != 0 and silent:
            print(
                f"pip install failed with error code {res.returncode}:", file=sys.stderr
            )
            print(res.stderr.decode(), file=sys.stderr)
    finally:
        os.unlink(f.name)
