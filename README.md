# The Neptyne Kernel

This package contains everything shipped in a [Neptyne](https://neptyne.com) kernel container. Running outside of a Neptyne context isnt fully supported yet, but this may prove useful for some advanced use cases.

## Installing

Neptyne kernels use Python 3.11, so this is the only version currently supported.

```python
python3.11 -m venv venv
. venv/bin/activate
pip install -r neptyne_kernel/requirements.txt # or pip install uv && uv pip install -r neptyne_kernel/requirements.txt
```

## Usage

In Neptyne containers, you'd typically import the API using

```
import neptyne as nt
```

Outside of this context, you'll need to do something like

```
import neptyne_kernel.neptyne_api as nt
```

### Local Kernels + Google Sheets

You can configure the Neptyne kernel to run in a Jupyter notebook, and connect it to a Google sheet. To do so, first install the Neptyne kernel spec:

```shell
neptyne_kernel_spec=$(python -c 'import neptyne_kernel, pathlib; print(pathlib.Path(neptyne_kernel.__file__).parent / "kernel_spec" / "neptyne")')
jupyter kernelspec install $neptyne_kernel_spec
```

Then, in a Jupyter notebook, select "Neptyne" as your kernel type. After obtaining an API key from a Google Sheet using the Neptyne extension, you can connect your notebook to your sheet using:

```python
import neptyne as nt

nt.connect_kernel("<api key>")
```

in your notebook.
