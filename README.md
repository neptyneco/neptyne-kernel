# The Neptyne Kernel

This package contains everything shipped in a [Neptyne](https://neptyne.com) kernel container.

## Python Version

Neptyne kernels use Python 3.11, so this is the only version currently tested/supported.

## Local Kernels + Google Sheets

You can use the Neptyne kernel to run in a Jupyter notebook, and connect it to a Google sheet. To do so, first get an API key from your Neptyne Google sheets extension. In the extension menu, open 'Manage Advanced Features', and enable the API. Copy your API key, and put in your notebook cell:


```python
import neptyne_kernel

neptyne_kernel.init_notebook("<your api key>")
```

You can now use Neptyne A1-notation in your notebook cells (e.g. `A1 = 1`)
