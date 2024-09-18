from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io.json
import pydeck as pdk
from IPython.core.display import HTML, DisplayObject
from plotly.basedatatypes import BaseFigure
from plotly.io import to_html
from pydeck.data_utils import compute_view

from ..neptyne_protocol import MIMETypes
from ..util import list_like
from .base_widget import (
    BaseWidget,
    ColorMixins,
    StrEnum,
    maybe_cast_to_list,
    widget_field,
)
from .register_widget import register_widget

if TYPE_CHECKING:
    pass

DEFAULT_OUTPUT_WIDGET_WIDTH = 600
DEFAULT_OUTPUT_WIDGET_HEIGHT = 400
MAX_OUTPUT_WIDGET_WIDTH = 1800
MAX_OUTPUT_WIDGET_HEIGHT = 1200

PLOTLY_MIME_TYPE = "application/vnd.plotly.v1+json"


plotly.io.json.config.default_engine = "json"


def plotly_to_html(fig_dict: dict, height: str = "100%") -> str:
    html = to_html(
        fig_dict,
        config={},
        auto_play=False,
        include_plotlyjs="cdn",
        include_mathjax=False,
        post_script=None,
        full_html=False,
        animation_opts=None,
        default_width="100%",
        default_height=height,
        validate=False,
    )
    return html


def plotly_to_image(fig_dict: dict, width: int, height: int, format: str) -> bytes:
    fig = BaseFigure(fig_dict)
    return fig.to_image(format=format, width=width, height=height)


def maybe_render_widget(
    value: Any, mime_bundle_to_update: dict[str, Any]
) -> DisplayObject | None:
    """Try to render value as a widget. Update mime_bundle_to_update with any alternate representation."""
    if isinstance(value, OutputWidget):
        value = value.render_widget()

    if isinstance(value, BaseFigure):
        margin = value.layout.margin
        for k in ("l", "r", "t", "b"):
            if margin[k] is None:
                margin[k] = 50 if k == "t" else 10  # Leave room for title
        value.update_layout(margin=margin)

        fig_dict = value.to_dict()
        mime_bundle_to_update[PLOTLY_MIME_TYPE] = fig_dict

        html = plotly_to_html(fig_dict)

        return HTML(html)

    if isinstance(value, pdk.Deck):
        return HTML(value.to_html(as_string=True, notebook_display=False))

    if isinstance(value, DisplayObject):
        return value

    return None


def transpose_rectangular_data(data: Any) -> Any:
    return [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))]


@dataclass(kw_only=True)
class OutputWidget(BaseWidget):
    mime_type = MIMETypes.APPLICATION_VND_NEPTYNE_OUTPUT_WIDGET_V1_JSON

    def render_widget(self) -> Any:
        pass


class PlotlyOutputWidget(OutputWidget):
    def render_figure(self) -> go.Figure:
        pass

    def render_widget(self) -> Any:
        return self.render_figure()


@register_widget(category="Output")
class Scatter(OutputWidget):
    """Use plotly to generate a scatter plot"""

    class TrendlineType(StrEnum):
        NONE = "none"
        LINEAR = "ols"
        POLY = "lowess"

    x: list[float] = widget_field("X data series")
    y: list[float] = widget_field("Y data series")
    radius: list[float] | None = widget_field("Size of each point", default=None)
    default_radius: float = widget_field("Default point size", default=5.0)
    max_radius: float = widget_field("Maximum point size", default=60.0)
    opacity: list[float] | None = widget_field("Opacity of each point", default=None)
    color: list[float | str] | None = widget_field("Color of each point", default=None)
    hover_labels: list[str] | None = widget_field(
        "Label on hover of each point", default=None
    )
    trendline: TrendlineType = widget_field(
        "Type of trendline", default=TrendlineType.NONE
    )
    log_x: bool = widget_field(
        "Render x on logarithmic scale?", default=False, inline=True
    )
    log_y: bool = widget_field(
        "Render y on logarithmic scale?", default=False, inline=True
    )
    title: str = widget_field("Plot title", category="Labels", default="", inline=True)
    x_label: str = widget_field(
        "X axis label", category="Labels", default="x", inline=True
    )
    y_label: str = widget_field(
        "Y axis label", category="Labels", default="y", inline=True
    )
    color_label: str = widget_field(
        "Color label", category="Labels", default="y", inline=True
    )
    animation_frame: list[float] | None = widget_field(
        "For animations what to key the frame to", default=None, category="Animation"
    )
    animation_group: list[str] | None = widget_field(
        "For animations what is animated", default=None, category="Animation"
    )
    kwargs: dict[str, Any] | None = widget_field(
        "Dictionary of args for plotly.express.scatter",
        default=None,
        category="Advanced",
    )

    def render_widget(self) -> go.Figure:
        radius = maybe_cast_to_list(self.radius)
        if not radius:
            radius = [self.default_radius for _ in self.x]

        args = dict(
            x=maybe_cast_to_list(self.x),
            y=maybe_cast_to_list(self.y),
            size=maybe_cast_to_list(radius),
            opacity=maybe_cast_to_list(self.opacity),
            color=maybe_cast_to_list(self.color),
            trendline=self.trendline.value
            if self.trendline.value != Scatter.TrendlineType.NONE.value
            else None,
            log_x=self.log_x,
            log_y=self.log_y,
            animation_frame=maybe_cast_to_list(self.animation_frame),
            animation_group=maybe_cast_to_list(self.animation_group),
            title=self.title,
            labels=dict(
                x=self.x_label,
                y=self.y_label,
                color=self.color_label,
            ),
            hover_name=maybe_cast_to_list(self.hover_labels),
            size_max=self.max_radius if radius else self.default_radius,
        )

        filtered_args = dict(filter(lambda item: item[1] is not None, args.items()))
        if self.kwargs:
            filtered_args = {**filtered_args, **self.kwargs}
        return px.scatter(**filtered_args)

    def validate_fields(self) -> dict[str, str]:
        field_errors = {}

        if len(self.x) == 0:
            field_errors["x"] = "X must have at least size 1"

        if len(self.y) == 0:
            field_errors["y"] = "Y must have at least size 1"

        if field_errors:
            return field_errors

        if len(self.x) != len(self.y):
            error_str = f"x and y must be the same size! X is size {len(self.x)}. Y is size {len(self.y)}"
            field_errors["x"] = error_str
            field_errors["y"] = error_str

        return field_errors


@register_widget(category="Output")
class Line(OutputWidget):
    """Show a line chart of a series of data"""

    class StackingType(StrEnum):
        NONE = ""
        OVERLAID = "overlaid"
        STACKED = "stacked"
        PERCENTAGE = "percentage"

    y: list[float] | list[list[float]] = widget_field("Y data series(s)")
    x: list[float] = widget_field("X axis data", default_factory=list)
    first_cell_is_header: bool = widget_field(
        "Data's first cell contains headers?",
        default=False,
        inline=True,
    )
    transpose_data: bool = widget_field(
        "Data specified in rows?", default=False, inline=True
    )
    smooth: bool = widget_field("Smooth the line?", default=False)
    title: str = widget_field("Plot title", category="Labels", default="")
    x_label: str = widget_field("X axis label", category="Labels", default="")
    y_label: str = widget_field("Y axis label", category="Labels", default="")
    first_series_as_bar: bool = widget_field(
        "Combo bar chart?", default=False, category="Advanced"
    )
    stacking: StackingType = widget_field(
        "Area stacking?", default=StackingType.NONE, category="Advanced"
    )

    def render_widget(self) -> go.Figure:
        fig = go.Figure()

        data: list[list[Any]]
        if list_like(self.y[0]):
            data = maybe_cast_to_list(self.y)
            if self.transpose_data:
                data = transpose_rectangular_data(data)
        else:
            data = [[float(v)] for v in self.y]  # type: ignore
        if self.stacking == Line.StackingType.PERCENTAGE:
            row_sums = [
                sum(row) for row in data[1 if self.first_cell_is_header else 0 :]
            ]
        else:
            row_sums = None
        x = maybe_cast_to_list(self.x)
        if not x:
            data_len = len(data)
            if self.first_cell_is_header:
                data_len -= 1
            x = [*range(data_len)]
        title = self.title
        labels = []
        line = {"shape": "spline", "smoothing": 1.3} if self.smooth else {}
        for col in range(len(data[0])):
            values = [row[col] for row in data]
            if self.first_cell_is_header:
                label, *values = values
                labels.append(label)
            else:
                label = f"Series {col}"
            if row_sums:
                values = [v / row_sums[row_idx] for row_idx, v in enumerate(values)]
            if self.first_series_as_bar and col == 0:
                trace = go.Bar(
                    name=label,
                    x=x,
                    y=values,
                )
            else:
                trace = go.Scatter(
                    x=x,
                    y=values,
                    name=label,
                    line=line,
                    fill="tonexty"
                    if self.stacking == Line.StackingType.OVERLAID
                    else None,
                    stackgroup="group"
                    if self.stacking
                    in (Line.StackingType.PERCENTAGE, Line.StackingType.STACKED)
                    else None,
                )
            fig.add_trace(trace)
        if title is None:
            title = ", ".join(labels)
        fig.update_layout(
            title=title,
            xaxis_title=self.x_label,
            yaxis_title=self.y_label,
        )
        return fig


@register_widget(category="Output")
class Column(OutputWidget):
    class BarMode(StrEnum):
        NORMAL = "normal"
        STACK = "stack"
        PERCENT = "percent"

    x: list[Any] = widget_field("X data series")
    y: list[float] | list[list[float]] = widget_field("Y data series(s)")
    first_cell_is_header: bool = widget_field(
        "Data's first cell contains headers?", default=False, inline=True
    )
    transpose_data: bool = widget_field(
        "Data specified in rows?", default=False, inline=True
    )

    title: str = widget_field("Plot title", category="Labels", default="")
    x_label: str = widget_field(
        description="X axis label", category="Labels", default="x"
    )
    y_label: str = widget_field(
        description="Y axis label", category="Labels", default="y"
    )

    mode: BarMode = widget_field(
        "The bar mode", default=BarMode.NORMAL, category="Advanced"
    )

    animation_frame: list[Any] | None = widget_field(
        description="Causes the rows of the data to be animated with this column as the frame",
        default=None,
        category="Advanced",
    )

    def get_orientation(self) -> str:
        return "v"

    def render_widget(self) -> go.Figure:
        mode = self.mode.value if hasattr(self.mode, "value") else self.mode

        data: list | np.ndarray

        animation_frame = self.animation_frame
        labels = None

        y = maybe_cast_to_list(self.y)
        x = maybe_cast_to_list(self.x)

        if y and list_like(y[0]):
            if animation_frame:
                y: list[list[float]] = maybe_cast_to_list(self.y)  # type: ignore
                if self.transpose_data:
                    y = transpose_rectangular_data(y)
                data = [x for row in y for x in row]
                llen = len(x)
                labels = [*x] * len(animation_frame)
                animation_frame = [animation_frame[i // llen] for i in range(len(data))]
            else:
                data = np.array(self.y)
                # We need to transpose for columns. Skip the transpose if data in rows.
                if not self.transpose_data:
                    data = data.T

                if mode == self.BarMode.PERCENT:
                    if self.first_cell_is_header:
                        sli = np.s_[:, 1:]
                    else:
                        sli = np.s_[:, :]

                    k = np.sum(data[sli], axis=0)
                    data[sli] = np.divide(data[sli], k) * 100
        else:
            if animation_frame:
                raise ValueError("Need two dimensional data for animation")
            data = [maybe_cast_to_list(y)]

        orientation = self.get_orientation()

        if animation_frame:
            if orientation == "h":
                data, labels = labels, data  # type: ignore
            fig = px.bar(
                x=labels,
                y=data,
                animation_frame=animation_frame,
                orientation=orientation,
            )
        else:
            plot_data = []
            for i, col in enumerate(data):
                if self.first_cell_is_header:
                    name = col[0]
                    y = col[1:]
                else:
                    name = f"Series {i}"
                    y = col
                x = self.x
                if orientation == "h":
                    x, y = y, x
                plot_data.append(go.Bar(x=x, y=y, name=name, orientation=orientation))

            fig = go.Figure(
                data=plot_data,
            )
            fig.update_layout(title=self.title)

        layout_args: dict[str, Any] = {}

        if mode == self.BarMode.STACK:
            layout_args["barmode"] = "stack"
        elif mode == self.BarMode.PERCENT:
            layout_args["barmode"] = "relative"
        elif mode == self.BarMode.NORMAL:
            pass
        else:
            raise ValueError("invalid bar mode: " + mode)

        fig.update_layout(
            **layout_args,
        )
        return fig


@register_widget(category="Output")
class Bar(Column):
    def get_orientation(self) -> str:
        return "h"


@register_widget(category="Output", enabled=False)
class TreeMap(OutputWidget):
    """Display hierarchical data in a tree map"""

    data: list[float] = widget_field(
        "The data series. Should include header",
    )
    level1: list[str] = widget_field(
        "Highest level categories (incl. header)",
    )
    level2: list[str] = widget_field(
        "secondary level categories (incl. header)",
    )
    summary_level: str | None = widget_field(
        "If set, summarize everything under this top level category",
        default="",
    )
    title: str = widget_field("Plot title", category="Labels", default="")
    kwargs: dict[str, Any] | None = widget_field(
        "Dictionary of args for plotly express Treemap",
        default=None,
        category="Advanced",
    )

    def render_widget(self) -> go.Figure:
        paths = [self.level1, self.level2]
        d = {lst[0]: lst[1:] for lst in [self.data, *paths]}  # type: ignore
        df = pd.DataFrame.from_dict(d)
        path_headers = [x[0] for x in paths]
        if self.summary_level:
            path_headers.insert(0, px.Constant(self.summary_level))

        args = dict(path=path_headers, values=self.data[0], title=self.title)
        if self.kwargs:
            args = {**args, **self.kwargs}

        return px.treemap(df, **args)


@register_widget(category="Output")
class Pie(OutputWidget):
    """Simple Pie Chart visualization"""

    data: list[float] = widget_field("Data to show in the pie chart")
    labels: list[str] = widget_field("Labels for each data entry", default_factory=list)
    donut: bool = widget_field("If true, punch the middle out", default=False)
    title: str = widget_field("Plot title", category="Labels", default="")
    kwargs: dict[str, Any] | None = widget_field(
        "Dictionary of args for plotly express Pie",
        default=None,
        category="Advanced",
    )

    def render_widget(self) -> go.Figure:
        labels = maybe_cast_to_list(self.labels)

        args = dict(
            values=maybe_cast_to_list(self.data),
            hole=0.3 if self.donut else 0,
            labels=labels or None,
            showlegend=bool(labels),
            title=self.title,
        )
        if self.kwargs:
            args = {**args, **self.kwargs}

        pie = go.Pie(**args)
        return go.Figure(data=[pie])


@register_widget(category="Output")
class Map(OutputWidget):
    """Renders a map of points based on their lat/long"""

    latitudes: list[float] = widget_field(
        "The latitude of each point",
    )

    longitudes: list[float] = widget_field(
        "The longitudes of each point",
    )

    radius: list[float] | None = widget_field(
        "The radius of each point in pixels",
        category="Rendering",
        default=None,
    )

    default_radius: float = widget_field(
        "The default radius of each point in pixels",
        category="Rendering",
        default=8.0,
        inline=True,
    )

    max_radius: float = widget_field(
        "The max radius of each point in pixels",
        category="Rendering",
        default=100.0,
        inline=True,
    )

    hover_labels: list[str] | None = widget_field(
        "Labels to show on hover for each point",
        category="Rendering",
        default=None,
    )

    labels: list[str] | None = widget_field(
        "Text labels to put on the map for each point",
        category="Rendering",
        default=None,
    )

    kwargs: dict[str, Any] | None = widget_field(
        "Dictionary of args for pydeck Layer",
        default=None,
        category="Advanced",
    )

    def render_widget(self) -> pdk.Deck:
        radius = maybe_cast_to_list(self.radius)
        labels = maybe_cast_to_list(self.labels)
        hover_labels = maybe_cast_to_list(self.hover_labels)
        radius = radius if radius else [self.default_radius for _ in self.latitudes]

        view = compute_view([list(c) for c in zip(self.longitudes, self.latitudes)])
        view.zoom -= 1  # zoom out because we are not full screen

        datadict: dict[str, Any] = {
            "lat": self.latitudes,
            "lng": self.longitudes,
            "radius": radius,
        }
        if labels:
            datadict["text"] = self.labels
        if hover_labels:
            datadict["caption"] = self.hover_labels
        df = pd.DataFrame(datadict)

        layer_args = dict(
            data=df,
            pickable=True,
            stroked=True,
            filled=True,
            radius_scale=1,
            radius_min_pixels=1,
            radius_max_pixels=self.max_radius,
            radius_units="'pixels'",
            line_width_min_pixels=1,
            get_position=["lng", "lat"],
            get_radius="radius",
            get_fill_color=[28, 154, 142],
            get_line_color=[0, 0, 0],
        )
        if self.kwargs:
            layer_args = {**layer_args, **self.kwargs}

        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                **layer_args,
            )
        ]

        if labels:
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=df,
                    pickable=True,
                    get_position=["lng", "lat"],
                    get_text="text",
                    get_size=10,
                )
            )

        view = compute_view([list(c) for c in zip(self.longitudes, self.latitudes)])
        view.zoom -= 2  # zoom out because we are not full screen

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view,
            map_style="light",
            tooltip={"text": "{caption}"} if hover_labels else True,
        )

        return deck


@register_widget(category="Output")
class Markdown(OutputWidget, ColorMixins):
    """Render Markdown text"""

    text: str = widget_field("The text to display", default="")

    def render_widget(self) -> HTML:
        try:
            import markdown
        except ImportError:
            raise ValueError(
                "You need to install the markdown package to use this widget"
            )

        background_color = (
            self.background_color.webcolor if self.background_color else "#ffffff"
        )
        text_color = self.text_color.webcolor if self.text_color else "#000000"
        style = (
            f"background-color: {background_color}; color: {text_color}; margin: 4px;"
        )
        style = "body { " + style + " }"

        return HTML(f"<style>{style}</style>" + markdown.markdown(self.text))


def Image(url: str, width: int | None = None, height: int | None = None) -> HTML:
    """Render an image from a url. We have two implementations, one sheets compatible and our own."""
    from ..formulas import IMAGE

    return IMAGE(url, 4, width, height)


@register_widget(category="Output", enabled=False)
class Container(OutputWidget):
    """Renders content in a container widget"""

    render_func: Callable | None = widget_field(
        "Function to render a widget", default=None
    )

    args: list = widget_field(
        "Arguments to pass to the render function",
        default_factory=list,
    )

    kwargs: dict = widget_field(
        "Keyword arguments to pass to the render function",
        default_factory=dict,
    )

    def render_widget(self) -> Any:
        return self.render_func(*self.args, **self.kwargs) if self.render_func else None
