"""
The sheets present in the spreadsheet modelled as Python collection.\n\n
Access with:\n\n
```python
import neptyne as nt
sheets = nt.sheets
```
In addition to the below methods, you can also access sheets by their name. For example:\n\n


```python
sheet = sheets['Sheet1'] # Get sheet by name\n
```\n\n

The sheet collection also supports iteration, so you can loop over all the sheets in the collection:\n\n

The returned sheet object is an instance of `NeptyneSheet` which is a subclass of `CellRange`
representing an infinite range in both dimensions.\n\n
"""

from ..sheet_api import NeptyneSheetCollection

__all__ = ["NeptyneSheetCollection"]
