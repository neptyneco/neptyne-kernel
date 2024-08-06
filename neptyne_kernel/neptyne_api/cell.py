"""# Neptyne's Cell API.
Cell ranges are the fundamental building block of Neptyne. They are the primary way to interact with data in a sheet.

Access to a cell's value is implicitly provided.

Set A1's value
```
A1 = 4
```

Retrieve A1's value
```
A1
```

Set A1's background color
```
A1.set_background_color('red')
```

For more high level context see [Cell API Overview](/neptyne_kernel/neptyne_api#cells)
"""

from ..cell_api import CellApiMixin

__all__ = ["CellApiMixin"]
