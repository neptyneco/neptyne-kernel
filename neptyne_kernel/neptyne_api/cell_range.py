"""CellRange is a class that represents a range of cells in a sheet. It can be 1D or 2D. CellRange is a subclass of Sequence\n\n
Of particular note is that cell ranges are iterable. This means that you can loop over the cells in a cell range.\n\n

For example:\n\n
```
for row in A1:C4:
    for cell in row:
        print(cell)
```

CellRanges are also indexable and sliceable. This means you can access individual cells or ranges of cells using square brackets.
This indexing supports numpy style syntax for convenience.\n\n

For example:\n\n

```
A1:C4[0] # Returns the first row of the cell range
A1:C4[0:2] # Returns the first 2 rows of the cell range
A1:C4[:, 1] # Returns the second column of the cell range
A1:C4[0, 1] # Returns the cell in the first row and second column

A1:C4[1, 0] = "Hello" # Sets the value of the cell in the second row and first column to "Hello"
A1:C4[1, 1:3] = [1, 2] # Sets the values of the cells in the first row and second and third columns to 1 and 2
A1:C4[0, 0] = ['spills', 'down', 'ward'] # Spills down the first column starting at the first row
```
"""

from ..cell_address import Range
from ..cell_range import CellRange

__all__ = ["CellRange", "Range"]
