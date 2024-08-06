from typing import Iterable

import plotly.graph_objects as go
from IPython.core.display import HTML
from plotly.io import to_html

from ..renderers import InlineWrapper


def Sparkline(data: Iterable[float]) -> InlineWrapper:
    data = [*data]
    fig = go.Figure(go.Scatter(x=list(range(len(data))), y=data, mode="lines"))
    if data[-1] > data[0]:
        fig.update_traces(line_color="green")
    else:
        fig.update_traces(line_color="red")
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return InlineWrapper(
        HTML(
            to_html(
                fig.to_dict(),
                config={"staticPlot": True},
                auto_play=False,
                include_plotlyjs="cdn",
                include_mathjax=False,
                post_script=None,
                full_html=False,
                animation_opts=None,
                default_width="100%",
                default_height="100%",
                validate=False,
            )
        )
    )
