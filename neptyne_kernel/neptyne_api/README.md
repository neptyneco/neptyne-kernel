# Neptyne API Docs: Welcome

Welcome to the Neptyne API docs! Here you'll find all the information you need to get started with Neptyne, as well as a
reference to all the available functions.

## First Steps

If you're new to Neptyne, you might want to check out
the [Quickstart Guide](https://www.neptyne.com/google-sheets-how-tos/quickstart) first.

It covers installation of the [Google Sheets Extension](TODO), an intro on the different parts of the UI and walks you
through
creating and running your first functions.

The tutorial series continues with more examples covering:

- [Data spilling](https://www.neptyne.com/google-sheets-how-tos/writing-your-own-functions)
- [Installing Python packages](https://www.neptyne.com/google-sheets-how-tos/using-python-pip-packages)
- [Calling external APIs with secrets](https://www.neptyne.com/google-sheets-how-tos/calling-apis-and-using-secrets).

## Py Function

The `Py()` function is the entry point for running Python code in your sheet.

Inside your editor panel create a hello world python function:

```python
def hello(to_who=None):
    if not to_who:
        to_who = 'world'
    return f'Hello, {to_who}!'
```

Then run it with the `Py()` function inside a sheets cell:

```=Py("hello")```

To pass parameters, try putting a string like "universe" in B2. Update the function call to:

```=Py("hello", B2)```

When B2 changes, notice that your Python code automatically re-runs and updates the cell with the new value.

## Advanced Features

By default, Neptyne for Sheets does not have programmatic access to your
spreadsheet data -- it can only see data passed into the `Py()` function, and the
only data written is the return value of `Py()`.

In order to read or write to your sheet from Python, you'll need to
enable [Advanced Features](https://www.neptyne.com/google-sheets/advanced-features).
This will have you authorize Neptyne to read and write data to your sheet. Don't worry - the
permission might look scary, but Neptyne's interface to Google Sheets makes sure that your code
can only modify the sheet it runs with.

Our tutorial series continues with examples that use Advanced Features:

- [Reading and writing values from Python](https://www.neptyne.com/google-sheets-how-tos/code-intro-neptyne)
- [Creating plots with Plotly](https://www.neptyne.com/google-sheets-how-tos/charts-with-plotly-v2)
- [Image Processing with PIL](https://www.neptyne.com/google-sheets-how-tos/google-sheets-images)

The rest of the API documentation assumes that you have enabled Advanced Features.

## Importing Neptyne API

The Neptyne API contains collection of functions that are available to use in your Neptyne sheets.

Generally, you'll want to have something like this at the top of your code panel:

```python
import neptyne as nt
```

Other behaviors are implicit in Neptyne, such as the ability to access cells and ranges as variables.

# Cells

The power of Neptyne comes from being able to access your spreadsheet's cells as python variables. The simplest way to
read and write to cells is with the standard cell syntax.

Set A1 to 3:

```python
A1 = 3
```

Read A1's value:

```python
>>> A1
3
```

This returns 3, but there's a little more to the story.

Try setting A1's background color to yellow:

```python
A1.set_background_color('yellow')
```

That's weird, it's not quite an integer since it has a `set_background_color` method. Let's check its type:

```python
>>> type(A1)
kernel.primitives.NeptyneInt
```

Aha! It's a `NeptyneInt`. Neptyne cells are objects that have methods to interact with the cell in the sheet.

However, you can still use them as regular integers in your code as they also have the same methods as the Python `int`
class.

Other proxied types exist on cells, such as `NeptyneFloat`, `NeptyneStr`.

Check the [Cell API](/neptyne_kernel/neptyne_api/cell#CellApiMixin) for a list of all the methods available on cells.

## Spilling Values

Neptyne automatically spills multi-dimensional values into neighboring cells.

For example, the following will set A1 to 1, A2 to 2, and A3 to 3.

```python
A1 = [1, 2, 3]
```

|          | A | B | C |
|----------|---|---|---
| <b>1</b> | 1 |   |   |
| <b>2</b> | 2 |   |   |
| <b>3</b> | 3 |   |   |
| <b>4</b> |   |   |   |

This means that 1D lists are spilled along columns. 2D lists are spilled by columns followed by rows.

```python
A1 = [[1, 2, 3], [4, 5, 6]]
```

|          | A | B | C |
|----------|---|---|---
| <b>1</b> | 1 | 2 | 3 |
| <b>2</b> | 4 | 5 | 6 |
| <b>3</b> |   |   |   |
| <b>4</b> |   |   |   |

Datatypes other than lists also support spilling. In particular dictionaries and pandas DataFrames are quite useful!

Dictionaries spill keys into the A column and values into the B column.

```python
>>> A1 = {"a": 1, "b": 2}
```

|          | A | B | C |
|----------|---|---|---
| <b>1</b> | a | 1 |   |
| <b>2</b> | b | 2 |   |
| <b>3</b> |   |   |   |
| <b>4</b> |   |   |   |

DataFrames spill column names into the A and the into subsequent rows

```python
>>> import pandas as pd
>>> A1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
```

|          | A | B | C |
|----------|---|---|---
| <b>1</b> | a | b |   |
| <b>2</b> | 1 | 2 |   |
| <b>3</b> | 3 | 4 |   |
| <b>4</b> |   |   |   |

### Performance

Generally, spilling should be used to assign values to many cells whenever possible. This only results in one call to
Google Sheets reducing the latency vs trying the same thing with a loop.

## Sheet Names

By default, A1 refers to A1 in your first (leftmost) sheet. You can access cells in other sheets by using the sheet
name.

For example, to set A1 in the sheet "Sheet2" to 3:

```
Sheet2!A1 = 3
```

# Cell Ranges

Cell ranges can be accessed similarly to cells with the standard range syntax.

Set the range A1:B2:

```python
A1:B2 = [[1, 2], [3, 4]]
```

|          | A | B | C |
|----------|---|---|---
| <b>1</b> | 1 | 2 |   |
| <b>2</b> | 3 | 4 |   |
| <b>3</b> |   |   |   |
| <b>4</b> |   |   |   |

Assignment and spilling works the same way as with cells, but the shape must match.

```python
>>> A1:B3 = [[1, 2], [3, 4]]
ValueError: Assigning ranges with different shapes (2, 2) vs (3, 2)
```

Get the range A1:B3:

```python
>>> A1:B3
CellRange: [[1, 2], [3, 4], [None, None]]
```

Cell ranges behave like 2 dimensional arrays, supporting indexing and slicing syntax

```python
>>> A1: B3[0]
CellRange: [1, 2]

>>> A1:B3[0, 0]
1

>>> A1:B3[0][0]
1

>>> A1: B3[0, :]
CellRange: [1, 2]

>>> A1:B3[:, 0]
CellRange: [1, 3, None]

>>> A1:B3[:, :]
CellRange: [[1, 2], [3, 4], [None, None]]
```

Cell ranges with only a single row or column behave like a 1D array. Regardless of if the range is a row or column, a
single index can be used to grab a value

```python
>>> A1: A3
CellRange: [1, 3, None]

>>> A1:C1
CellRange: [1, 2, None]

>>> A1:C1[1]
2

>>> A1:A3[0:2]
CellRange: [1, 3]
```

## Cell Range API

Cell ranges support the full range of methods in the [Cell API](/kerneliMixin), as well as some additional
methods for working with ranges.

For example the following will set the background color of all cells in the range A1:B2 to yellow.

```python
A1:B2.set_background_color('yellow')
```

Similarly, like cells you can specify the sheet name to access ranges in other sheets.

```
Sheet2!A1:B2
```

Cell ranges also have additional API methods.

In particular ```to_dataframe()``` comes in handy for working with popular python packages.

```to_list()``` can also be quite useful. While cell ranges act like 2D arrays some packages have specific checks for
lists.

```python
>>> df = A1:B3.to_dataframe()
>>> l = A1:B3.to_list()
```

A full list of available functions for [Cell Range](/neptyne_kernel/neptyne_api/cell_range#CellRange) is here.

## Insert and delete API

Of particular utility, are the insert and delete methods for cell ranges:

- [insert_row](/neptyne_kernel/neptyne_api/cell_range#CellRange.insert_row)
- [delete_row](/neptyne_kernel/neptyne_api/cell_range#CellRange.delete_row)
- [append_row](/neptyne_kernel/neptyne_api/cell_range#CellRange.append_row)

These operations are bound to the cell range they are called on and won't read or write to any other cells.

Here are a few examples:

Insert an empty row before the first row in the range A1:B2:

```python
>>> A1:B2.insert_row(0)
```

Delete the second row (index 1) in the range A1:D4:

```python
>>> A1:D4.delete_row(1)
```

## Infinite ranges

You can also access infinite ranges as cell ranges in both directions using syntax similar to regular ranges.

For example, ```A1:B``` has an infinite number of rows, and ```C5:8``` has an infinite number of columns.

These behave largely the same as regular ranges, but are infinite in the direction that they are specified.
They is particularly useful for appending data to the end of a sheet,
or other use cases with a growing or indeterminate size range.

```python
    A1:B.append_row(["new", "data"])
```

# Sheets

You can access cell ranges for entire sheets using the nt.sheets collection.

```python
nt.sheets['Sheet1']
```

This expression returns a cell range object that represents the entire sheet.
You can use this object to access the entire sheet as a 2D array.
This sheet cell range is also infinite in both directions.

All of the methods available for cell ranges are also available for sheet ranges. For example:

```python
nt.sheets['Sheet1'].insert_row(0, ['New text', 'To appear', 'At the top of my sheet'])
```

## Accessing cells via indexes

Another common use case for the sheets collection is to have more programmatic access to getting and setting certain cells.

For example, if I want to access the cell below B3 in Sheet1, I can do the following:

```python
x, y = B3.xy
nt.sheets['Sheet1'][x, y + 1] = 'Below B3'
```

## Inserting and deleting sheets

[NeptyneSheet](/neptyne_kernel/neptyne_api/sheet#NeptyneSheetCollection) has a full list of available functions
available on the sheet collection such as programmatically adding new sheets with:

```python
nt.sheets.new_sheet('New Sheet')
```

# Streamlit

Neptyne for Sheets supports building [Streamlit](https://streamlit.io/) apps. Streamlit is a popular Python library for
building interactive data-driven web apps.
You can use the `nt.streamlit` decorator to define a
Streamlit app in your code panel.

> **Note:** You'll almost certainly want to enable Advanced Features to make the
> most of a Streamlit app. Streamlit apps run on your backend, so in order to read or
> write any data from/to the sheet, you'll need to authorize [Advanced Features](#advanced-features).

```python
import neptyne as nt
import streamlit as st


@nt.streamlit
def app():
    st.title('My Streamlit App')
    st.write('Hello, world!')
```

## Pop-up vs Sidebar

By default, Streamlit apps are displayed in a pop-up window.
You can change this to a sidebar by passing `sidebar=True`
to the `nt.streamlit` decorator.

```python
import neptyne as nt
import streamlit as st


@nt.streamlit(sidebar=True)
def app():
    if st.button("Refresh Data"):
        A1 = get_data()
```

In pop-up mode, you can also set `width` and `height`.

## Auto-Open

If you want your Streamlit app to open automatically when the sheet is loaded,
you can pass `auto_open=True` to the `nt.streamlit` decorator.

```python
import neptyne as nt
import streamlit as st


@nt.streamlit(auto_open=True)
def app():
    st.write('Who goes there?')
    name = st.text_input('Enter your name')
    A1: A.append_row([f"{name} says hello!"])
```

# Secrets

Secrets are managed primarily through the Neptyne UI. You can access them in your code via the
[get_secret](/neptyne_kernel/neptyne_api#get_secret) function.

# Full Google API Access

Currently Neptyne is under active development and we are working on adding more features and improving our API.

If we are missing a features, it is possible to access the full Google API using the [google](/neptyne_kernel/neptyne_api/google) module.

We've made things much easier for you though! In particular all of the authentication and setup is done for you.
Additionally, Neptyne cell ranges can be used as a proxy for Google Sheets ranges.

For example, to copy paste from A1:A4 -> B1:B4:

```python
google.sheets.copy_paste(A1:A4, B1:B4)
```

# Other API Functions

The rest of the API documentation on this page covers useful functions that are available to you in your Neptyne code.

Remember to import the neptyne library at the top of your code panel and access these functions with nt.:

```python
import neptyne as nt
```
