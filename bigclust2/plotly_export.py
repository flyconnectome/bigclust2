from numbers import Number

import numpy as np
import pandas as pd

# Mapping from pygfx MarkerShape names to the closest Plotly symbol names.
# pygfx "plus" is a + shape; Plotly calls that "cross".
# pygfx "cross" is an × shape; Plotly calls that "x".
_PYGFX_TO_PLOTLY_SYMBOL = {
    "circle": "circle",
    "ring": "circle-open",
    "square": "square",
    "diamond": "diamond",
    "plus": "cross",
    "cross": "x",
    "asterix": "asterisk",
    "tick": "line-ns",
    "tick_left": "line-ew",
    "tick_right": "line-ew",
    "triangle_up": "triangle-up",
    "triangle_down": "triangle-down",
    "triangle_left": "triangle-left",
    "triangle_right": "triangle-right",
    "heart": "star-triangle-up",
    "spade": "star-triangle-down",
    "club": "star-square",
    "pin": "star-diamond",
    "custom": "circle",
}


def _resolve_marker_symbols(scatter):
    """Return a per-point array of Plotly symbol names for *scatter*.

    Uses the already-computed ``_marker_symbols`` attribute when available
    (populated by ``ScatterFigure.make_points``), otherwise derives it from
    ``scatter.markers`` using the same logic.
    """
    from .scatter import AVAILABLE_MARKERS

    if getattr(scatter, "_marker_symbols", None) is not None:
        gfx_symbols = scatter._marker_symbols
    elif scatter.markers is None:
        return None  # all circles, let Plotly use its default
    else:
        unique_types = np.unique(scatter.markers)
        marker_map = dict(zip(unique_types, AVAILABLE_MARKERS))
        gfx_symbols = np.array([marker_map[t] for t in scatter.markers])

    plotly_symbols = np.array(
        [_PYGFX_TO_PLOTLY_SYMBOL.get(s, "circle") for s in gfx_symbols]
    )

    # If every point uses the same symbol, return a scalar string instead of
    # an array so Plotly can apply it more efficiently.
    unique = np.unique(plotly_symbols)
    if len(unique) == 1:
        return unique[0]
    return plotly_symbols


def scatter_to_plotly_figure(
    scatter,
    include_selected=True,
    selected_size=10,
    unselected_opacity=0.85,
    marker_line_width=0,
):
    """Export a scatter figure to a Plotly scatter figure.

    Parameters
    ----------
    scatter : ScatterFigure
        Source scatter figure instance.
    include_selected : bool, optional
        If True, selected points are rendered as a separate highlighted trace.
    selected_size : int, optional
        Marker size used for the selected-points highlight trace.
    unselected_opacity : float, optional
        Opacity of the main scatter trace.
    marker_line_width : float, optional
        Outline width for markers in both traces.
    """
    if scatter.positions is None or len(scatter) == 0:
        raise ValueError("No points available to export.")

    if scatter.positions.ndim != 2 or scatter.positions.shape[1] < 2:
        raise ValueError(
            f"Expected positions with shape (N, 2+) for export, got {scatter.positions.shape}."
        )

    # Delayed import keeps Plotly out of the default import path.
    import plotly.graph_objects as go

    x = scatter.positions[:, 0]
    y = scatter.positions[:, 1]

    hover_text = None
    if getattr(scatter, "hover_info", None) is not None:
        hover_text = np.array(
            [
                "<br>".join(scatter._format_hover_text(v).splitlines())
                for v in scatter.hover_info
            ]
        )

    if isinstance(scatter.point_size, Number):
        marker_size = float(scatter.point_size) * float(scatter.point_scale)
    else:
        marker_size = (
            np.asarray(scatter.point_size, dtype=np.float32)
            * float(scatter.point_scale)
        )

    plotly_symbols = _resolve_marker_symbols(scatter)

    marker = {
        "size": marker_size,
        "sizemin": 2,
        "line": {"width": marker_line_width},
    }
    if plotly_symbols is not None:
        marker["symbol"] = plotly_symbols

    if scatter.colors is not None:
        if isinstance(scatter.colors, np.ndarray) and scatter.colors.ndim == 2:
            marker["color"] = [
                f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})"
                for r, g, b, a in scatter.colors
            ]
        else:
            marker["color"] = np.asarray(scatter.colors)

    customdata = None
    hovertemplate = "%{text}<extra></extra>"
    if scatter.metadata is not None:
        customdata = scatter.metadata.to_dict("records")

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=marker,
            opacity=unselected_opacity,
            text=hover_text,
            customdata=customdata,
            hovertemplate=hovertemplate if hover_text is not None else None,
            name="points",
        )
    )

    if include_selected and scatter.selected is not None and len(scatter.selected):
        sel = np.asarray(scatter.selected, dtype=int)
        sel_hover = hover_text[sel] if hover_text is not None else None

        # Use the same symbols as the underlying points but in the open variant
        # so the selection ring does not obscure the filled point beneath it.
        if plotly_symbols is not None and not isinstance(plotly_symbols, str):
            sel_symbols = np.array(
                [s + "-open" if not s.endswith("-open") else s for s in plotly_symbols[sel]]
            )
        elif isinstance(plotly_symbols, str):
            sel_symbols = plotly_symbols + "-open" if not plotly_symbols.endswith("-open") else plotly_symbols
        else:
            sel_symbols = "circle-open"

        fig.add_trace(
            go.Scattergl(
                x=x[sel],
                y=y[sel],
                mode="markers",
                marker={
                    "size": selected_size,
                    "symbol": sel_symbols,
                    "color": "rgba(255, 255, 0, 0.95)",
                    "line": {"width": max(1.0, marker_line_width + 1)},
                },
                text=sel_hover,
                hovertemplate=hovertemplate if hover_text is not None else None,
                name="selected",
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis={"title": "Embedding 1", "zeroline": False},
        yaxis={"title": "Embedding 2", "zeroline": False, "scaleanchor": "x"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )

    return fig


def _build_dashboard_data(scatter, top_n=20, include_selected=True):
    """Build the dashboard figure and return metadata for JS cross-filtering.

    Returns
    -------
    tuple of (figure, datasets_per_point, labels_per_point,
              dataset_trace_idx, label_trace_idx)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    base_fig = scatter_to_plotly_figure(scatter, include_selected=include_selected)
    n_scatter_traces = len(base_fig.data)

    dashboard = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        column_widths=[0.72, 0.28],
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=("Embedding", "Datasets", f"Labels (top {top_n})"),
    )

    for trace in base_fig.data:
        dashboard.add_trace(trace, row=1, col=1)

    trace_idx = n_scatter_traces
    dataset_trace_idx = -1
    label_trace_idx = -1
    datasets_per_point = None
    labels_per_point = None

    metadata = getattr(scatter, "metadata", None)
    if metadata is not None and len(metadata):
        if "dataset" in metadata.columns:
            dataset_counts = metadata["dataset"].astype(str).value_counts().head(top_n)
            dashboard.add_trace(
                go.Bar(
                    x=dataset_counts.values,
                    y=dataset_counts.index,
                    orientation="h",
                    name="datasets",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            dataset_trace_idx = trace_idx
            trace_idx += 1
            datasets_per_point = metadata["dataset"].astype(str).tolist()

        if "label" in metadata.columns:
            label_counts = metadata["label"].astype(str).value_counts().head(top_n)
            labels_per_point = metadata["label"].astype(str).tolist()
        elif getattr(scatter, "labels", None) is not None:
            label_counts = pd.Series(np.asarray(scatter.labels, dtype=str)).value_counts().head(top_n)
            labels_per_point = np.asarray(scatter.labels, dtype=str).tolist()
        else:
            label_counts = None

        # Reverse label_counts so most frequent appears at the top
        if label_counts is not None:
            label_counts = label_counts[::-1]

        if label_counts is not None:
            dashboard.add_trace(
                go.Bar(
                    x=label_counts.values,
                    y=label_counts.index,
                    orientation="h",
                    name="labels",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
            label_trace_idx = trace_idx
            trace_idx += 1

    dashboard.update_layout(
        template="plotly_white",
        height=900,
        title_text="Embedding Dashboard",
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
    )
    dashboard.update_xaxes(title_text="Embedding 1", row=1, col=1)
    dashboard.update_yaxes(title_text="Embedding 2", row=1, col=1, scaleanchor="x")
    dashboard.update_xaxes(title_text="Count", row=1, col=2)
    dashboard.update_xaxes(title_text="Count", row=2, col=2)

    return dashboard, datasets_per_point, labels_per_point, dataset_trace_idx, label_trace_idx


def scatter_to_dashboard_figure(
    scatter,
    top_n=20,
    include_selected=True,
):
    """Create a compact multi-panel dashboard figure for a ScatterFigure.

    Panels:
    - Main embedding scatter
    - Dataset counts bar chart
    - Label counts bar chart (top N)
    """
    fig, *_ = _build_dashboard_data(scatter, top_n=top_n, include_selected=include_selected)
    return fig


def _build_selection_js(
    datasets_per_point, labels_per_point, dataset_trace_idx, label_trace_idx, top_n
):
    """Return a self-contained JS snippet that cross-filters bar panels on scatter selection."""
    import json

    ds_json  = json.dumps(datasets_per_point) if datasets_per_point  is not None else "null"
    lbl_json = json.dumps(labels_per_point)   if labels_per_point    is not None else "null"

    return f"""
(function() {{
    var _ds  = {ds_json};
    var _lbl = {lbl_json};
    var _dsIdx  = {dataset_trace_idx};
    var _lblIdx = {label_trace_idx};
    var _topN   = {top_n};

    function countTop(arr) {{
        var counts = {{}};
        for (var i = 0; i < arr.length; i++) {{
            var v = String(arr[i]);
            if (v && v !== 'null' && v !== 'None' && v !== 'nan') {{
                counts[v] = (counts[v] || 0) + 1;
            }}
        }}
        var entries = Object.entries(counts)
            .sort(function(a, b) {{ return b[1] - a[1]; }})
            .slice(0, _topN);
        return {{
            names:  entries.map(function(e) {{ return e[0]; }}),
            values: entries.map(function(e) {{ return e[1]; }})
        }};
    }}

    // applyBars: indices=null means show full dataset (deselect / empty selection)
    function applyBars(gd, indices) {{
        if (_dsIdx >= 0 && _ds) {{
            var arr = (indices !== null) ? indices.map(function(i) {{ return _ds[i]; }}) : _ds;
            var c = countTop(arr);
            Plotly.update(gd,
                {{'x': [c.values], 'y': [c.names]}},
                {{'xaxis2.autorange': true, 'yaxis2.autorange': true}},
                [_dsIdx]
            );
        }}
        if (_lblIdx >= 0 && _lbl) {{
            var arr = (indices !== null) ? indices.map(function(i) {{ return _lbl[i]; }}) : _lbl;
            var c = countTop(arr);
            Plotly.update(gd,
                {{'x': [c.values], 'y': [c.names]}},
                {{'xaxis3.autorange': true, 'yaxis3.autorange': true}},
                [_lblIdx]
            );
        }}
    }}

    window.addEventListener('load', function() {{
        var gd = document.querySelector('.js-plotly-plot');
        if (!gd) return;

        gd.on('plotly_selected', function(eventData) {{
            if (!eventData || !eventData.points || eventData.points.length === 0) {{
                applyBars(gd, null);
                return;
            }}
            // Only react to the main scatter trace (curveNumber 0)
            var pts = eventData.points.filter(function(p) {{ return p.curveNumber === 0; }});
            if (pts.length === 0) {{ applyBars(gd, null); return; }}
            applyBars(gd, pts.map(function(p) {{ return p.pointIndex; }}));
        }});

        gd.on('plotly_deselect', function() {{ applyBars(gd, null); }});
    }});
}})();
"""


def write_scatter_dashboard_html(
    scatter,
    file_path,
    top_n=20,
    include_selected=True,
    include_plotlyjs="cdn",
):
    """Write a standalone multi-panel dashboard HTML file.

    The exported file includes JavaScript so that box/lasso-selecting points in
    the scatter panel updates the Datasets and Labels bar charts to reflect the
    composition of the selection.  Deselecting restores the full counts.
    """
    fig, datasets_per_point, labels_per_point, dataset_trace_idx, label_trace_idx = (
        _build_dashboard_data(scatter, top_n=top_n, include_selected=include_selected)
    )

    html = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)

    if dataset_trace_idx >= 0 or label_trace_idx >= 0:
        js = _build_selection_js(
            datasets_per_point, labels_per_point,
            dataset_trace_idx, label_trace_idx, top_n,
        )
        html = html.replace("</body>", f"<script>\n{js}\n</script>\n</body>", 1)

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return fig

