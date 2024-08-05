# The Neptyne Kernel

This package contains everything shipped in a [Neptyne](https://neptyne.com) kernel container. Running outside of a Neptyne context isnt fully supported yet, but this may prove useful for some advanced use cases.

## Installing

Neptyne kernels use Python 3.11, so this is the only version currently supported.

python```
python3.11 -m venv venv
. venv/bin/activate
pip install -r kernel/requirements.txt # or pip install uv && uv pip install -r kernel/requirements.txt
```

In Neptyne containers, you'd typically import the API using

```
import neptyne as nt
```

Outside of this context, you'll need to do something like

```
import kernel.neptyne_api as nt
```
