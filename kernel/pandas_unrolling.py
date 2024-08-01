from typing import Any

import pandas as pd


def dataframe_to_grid(df: pd.DataFrame) -> list[list[Any]]:
    df_header = [*df]
    if (
        isinstance(df.index, pd.RangeIndex)
        and df.index.start == 0
        and df.index.step == 1
    ):
        include_index = False
    else:
        include_index = True
        df_header.insert(0, get_pandas_index_name(df))
    return [[clean_header(v) for v in df_header]] + [
        list(t) for t in df.itertuples(include_index)
    ]


def get_pandas_index_name(df: pd.DataFrame) -> str:
    # Both the index and the columns can have a name. One way to do this is to have
    # a named index, do a transpose, and then set a named index on the result.
    # We only have one cell to use so if both exist, we end up only unrolling the
    # index name.
    for ix in (df.index, df.columns):
        # Indexes have "name", and MultiIndexes have "names"
        names = getattr(ix, "names", [None])
        if len(names) == 1 and names[0] is None:
            continue
        return names if len(names) > 1 else names[0]
    return ""


def clean_header(header: tuple | Any) -> Any:
    if isinstance(header, str):
        return header
    if not hasattr(header, "__len__"):
        return header
    if len(header) == 1:
        return header[0]
    for h in header[1:]:
        # Collapse multi-indexes when the name repeats, when happens a lot with tables from HTML
        if h != header[0]:
            return " ".join(map(str, header))
    return header[0]
