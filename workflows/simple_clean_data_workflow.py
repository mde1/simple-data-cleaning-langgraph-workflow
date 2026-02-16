from typing_extensions import TypedDict, Literal
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pathlib import Path
import io
import streamlit as st
import tempfile
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# Load environment variables from .env file
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------
# 1. Shared State Definition
# ---------------------------

class DataState(TypedDict, total=False):
    csv_path: str

    # Dataframes
    df_before: pd.DataFrame
    df_after: pd.DataFrame

    # Audit tables
    missing_rows_before: pd.DataFrame           # rows that had missing values originally
    outlier_cells_before: pd.DataFrame          # row/col/value for detected outliers (before)

    # Summaries
    summary_before: dict
    summary_after: dict

    # Visualizations
    viz_plan: dict
    viz_reason: str

    # text cleaning
    text_action: Literal["lowercase", "tokenize", "both", "none"]
    text_columns: list[str]

    # Decision
    action: Literal["clean_missing", "remove_outliers", "both", "none"]


# ---------------------------
# 2. Initialize LLM
# ---------------------------

llm = ChatOpenAI(model="gpt-4o-mini")


# ---------------------------
# 3. Nodes
# ---------------------------

def load_data(state: DataState) -> DataState:
    df = pd.read_csv(state["csv_path"])
    state["df_before"] = df
    return state

def summarize_before(state: DataState) -> DataState:
    df = state["df_before"]
    state["summary_before"] = make_summary(df)

    # rows that had missing values originally
    missing_mask = df.isna().any(axis=1)
    state["missing_rows_before"] = df.loc[missing_mask].copy()

    # outlier cells (before)
    _, outlier_tbl = detect_outliers_iqr(df)
    state["outlier_cells_before"] = outlier_tbl

    return state


def summarize_data(state: DataState) -> DataState:
    """Generate comprehensive summary including missing values."""
    parts = []
    
    # 1. Basic statistics
    parts.append("DATA DESCRIPTION:\n")
    parts.append(state["df"].describe().to_string())
    
    # 2. Dataset info
    parts.append("\n\nDATA INFO:\n")
    buf = io.StringIO()
    state["df"].info(buf=buf)
    parts.append(buf.getvalue())
    
    # 3. Explicit missing value counts
    parts.append("\n\nMISSING VALUE COUNTS:\n")
    missing_counts = state["df"].isnull().sum()
    parts.append(missing_counts.to_string())
    
    state["summary"] = "\n".join(parts)
    return state

def apply_cleaning(state: DataState) -> DataState:
    df = state["df_before"].copy()
    action = state.get("action", "none")

    if action == "clean_missing":
        df = fill_missing_means(df)
    elif action == "remove_outliers":
        df = winsorize_iqr(df)
    elif action == "both":
        # winsorize first, then fill missing (or reverse—either is defensible)
        df = winsorize_iqr(df)
        df = fill_missing_means(df)

    state["df_after"] = df
    return state

def apply_text_processing(state: DataState) -> DataState:
    df = state["df_after"].copy()
    action = state.get("text_action", "none")
    columns = state.get("text_columns", [])

    if action == "lowercase":
        df = lowercase_text_columns(df, columns if columns else None)

    elif action == "tokenize":
        df = tokenize_columns(df, columns)

    elif action == "both":
        df = lowercase_text_columns(df, columns if columns else None)
        df = tokenize_columns(df, columns)

    state["df_after"] = df
    return state

def summarize_after(state: DataState) -> DataState:
    state["summary_after"] = make_summary(state["df_after"])
    return state


def output_state(state: DataState) -> DataState:
    # no printing in Streamlit; keep node for graph completeness
    return state


def reasoning_node(state: DataState) -> DataState:
    """Use LLM to decide whether to clean missing values or remove outliers."""
    df = state["df_before"]
    summ = state["summary_before"]

    # Lightweight text prompt (avoid dumping huge tables)
    top_missing = summ["missing_by_col"].head(10).to_string()
    numeric_desc = summ["numeric_describe"].head(10).to_string() if not summ["numeric_describe"].empty else "None"

    prompt = f"""
            You are a data science assistant.
            Decide the best single action: clean_missing, remove_outliers, both, or none.

            Constraints:
            - Do NOT drop any rows or columns.
            - "remove_outliers" means winsorize/clip extreme numeric values (IQR), not row removal.
            - "clean_missing" means fill missing numeric values with column mean.

            Dataset:
            - shape: {summ["shape"]}
            - total_missing: {summ["missing_total"]}

            Top missing by column:
            {top_missing}

            Sample numeric describe (first 10 numeric columns):
            {numeric_desc}

            Respond ONLY with one of: clean_missing, remove_outliers, both, none.
            """.strip()

    decision = llm.invoke(prompt).content.strip().lower()
    if decision not in ["clean_missing", "remove_outliers", "both"]:
        decision = "none"
    state["action"] = decision
    return state

def suggest_visualizations(state: DataState) -> DataState:
    # Choose whether you want viz on BEFORE, AFTER, or BOTH
    # Most people want AFTER (cleaned) + a missingness plot from BEFORE.
    df_for_viz = state["df_after"] if "df_after" in state else state["df_before"]

    df_for_viz_choice = state["df_before"]
    plan = choose_viz_plan_with_llm(df_for_viz_choice, llm)

    state["viz_plan"] = plan
    state["viz_reason"] = plan.get("reason", "")
    return state

# ---------------------------
# 7.1. Visualization Helpers
# ---------------------------
def build_viz_candidates(df: pd.DataFrame, max_categories: int = 30) -> list[dict]:
    nrows, ncols = df.shape
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    # basic stats for selection
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    top_missing_cols = missing_by_col[missing_by_col > 0].head(5).index.tolist()

    # cardinality (for categories)
    cat_card = {c: int(df[c].nunique(dropna=True)) for c in cat_cols}
    low_card_cats = [c for c in cat_cols if 2 <= cat_card.get(c, 0) <= max_categories]

    candidates = []

    # Always useful
    candidates.append({"id": "missing_bar", "type": "bar_missing_by_col"})
    if len(num_cols) >= 2:
        candidates.append({"id": "corr_heatmap", "type": "heatmap_corr"})

    # Numeric distributions
    for c in num_cols[:8]:
        candidates.append({"id": f"hist_{c}", "type": "hist", "x": c})
        candidates.append({"id": f"box_{c}", "type": "box", "y": c})

    # Category counts
    for c in low_card_cats[:6]:
        candidates.append({"id": f"count_{c}", "type": "count", "x": c})

    # Scatter candidates: pick a few pairs
    if len(num_cols) >= 2:
        pairs = []
        for i in range(min(3, len(num_cols))):
            for j in range(i + 1, min(4, len(num_cols))):
                pairs.append((num_cols[i], num_cols[j]))
        for x, y in pairs[:4]:
            candidates.append({"id": f"scatter_{x}_{y}", "type": "scatter", "x": x, "y": y})

    # If datetime present, timeseries against a numeric column
    if dt_cols and num_cols:
        candidates.append({"id": "timeseries", "type": "timeseries", "x": dt_cols[0], "y": num_cols[0]})

    # Missing rows preview (table isn’t plotly, but you already show it)
    if top_missing_cols:
        candidates.append({"id": "missing_matrix_light", "type": "missingness_light", "cols": top_missing_cols})

    return candidates


def choose_viz_plan_with_llm(df: pd.DataFrame, llm: ChatOpenAI) -> dict:
    candidates = build_viz_candidates(df)

    # Keep prompt small
    nrows, ncols = df.shape
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    missing_total = int(df.isna().sum().sum())

    prompt = f"""
        You are a data visualization assistant.
        Pick the best 3–6 visualizations from the candidates list to help a user understand this dataset quickly.

        Constraints:
        - Use ONLY candidate IDs provided.
        - Prefer clarity over quantity.
        - If there are missing values, include missing_bar.
        - If there are 2+ numeric columns, correlation heatmap is usually helpful.

        Dataset:
        - shape: ({nrows}, {ncols})
        - numeric_cols: {num_cols[:15]}
        - categorical_cols: {cat_cols[:15]}
        - total_missing: {missing_total}

        Candidates (JSON):
        {json.dumps(candidates)}


        Return STRICT JSON with this schema:
        {{
        "selected": ["candidate_id_1", "candidate_id_2", ...],
        "reason": "1-3 sentences explaining why these are best"
        }}
        """.strip()

    raw = llm.invoke(prompt).content.strip()

    # Guardrail parse
    try:
        plan = json.loads(raw)
        selected = plan.get("selected", [])
        selected_set = {c["id"] for c in candidates}
        selected = [s for s in selected if s in selected_set]
        if not selected:
            selected = [candidates[0]["id"]]  # fallback
        return {"selected": selected[:6], "reason": plan.get("reason", "")}
    except Exception:
        # Safe fallback
        fallback = []
        ids = {c["id"] for c in candidates}
        if "missing_bar" in ids: fallback.append("missing_bar")
        if "corr_heatmap" in ids: fallback.append("corr_heatmap")
        # add one hist if exists
        for c in candidates:
            if c["type"] == "hist":
                fallback.append(c["id"])
                break
    return {"selected": fallback[:3], "reason": "Fallback selection (LLM output was not valid JSON)."}

def make_paired_figures(df_before: pd.DataFrame, df_after: pd.DataFrame, cand: dict):
    """
    Returns (fig_before, fig_after, fig_overlay_or_none)
    Overlay is only used when it makes sense (e.g., hist/box/scatter).
    """
    t = cand["type"]

    # Keep overlays fast for very large data
    def maybe_sample(df, kind):
        if len(df) > 200_000 and kind in {"scatter", "hist"}:
            return df.sample(50_000, random_state=42)
        return df

    b = maybe_sample(df_before, t)
    a = maybe_sample(df_after, t)

    overlay = None

    if t == "bar_missing_by_col":
        miss_b = df_before.isna().sum().sort_values(ascending=False)
        miss_a = df_after.isna().sum().sort_values(ascending=False)

        miss_b = miss_b[miss_b > 0].head(30)
        miss_a = miss_a[miss_a > 0].head(30)

        plot_b = miss_b.reset_index()
        plot_b.columns = ["column", "missing_count"]

        plot_a = miss_a.reset_index()
        plot_a.columns = ["column", "missing_count"]


        fig_b = px.bar(
            plot_b, 
            x="column", 
            y="missing_count",
            title="Before: Missing values (top 30)"
                        )

        fig_a = px.bar(
            plot_a, 
            x="column",
            y="missing_count",
            title="After: Missing values (top 30)"
        )


        fig_b.update_layout(xaxis_tickangle=-45)
        fig_a.update_layout(xaxis_tickangle=-45)
        return fig_b, fig_a, None

    if t == "heatmap_corr":
        num_b = df_before.select_dtypes(include="number")
        num_a = df_after.select_dtypes(include="number")
        if num_b.shape[1] < 2 or num_a.shape[1] < 2:
            return None, None, None
        corr_b = num_b.corr(numeric_only=True)
        corr_a = num_a.corr(numeric_only=True)
        fig_b = px.imshow(corr_b, title="Before: Correlation heatmap")
        fig_a = px.imshow(corr_a, title="After: Correlation heatmap")
        return fig_b, fig_a, None

    if t == "hist":
        x = cand["x"]
        fig_b = px.histogram(b, x=x, nbins=40, title=f"Before: Histogram of {x}")
        fig_a = px.histogram(a, x=x, nbins=40, title=f"After: Histogram of {x}")

        # Overlay (same bins visually; we’ll just overlay counts)
        overlay = go.Figure()
        overlay.add_trace(go.Histogram(x=b[x], nbinsx=40, name="Before", opacity=0.55))
        overlay.add_trace(go.Histogram(x=a[x], nbinsx=40, name="After", opacity=0.55))
        overlay.update_layout(
            barmode="overlay",
            title=f"Overlay: {x} (Before vs After)",
            xaxis_title=x,
            yaxis_title="Count",
        )
        return fig_b, fig_a, overlay

    if t == "box":
        y = cand["y"]
        fig_b = px.box(b, y=y, points="outliers", title=f"Before: Box plot of {y}")
        fig_a = px.box(a, y=y, points="outliers", title=f"After: Box plot of {y}")

        overlay = go.Figure()
        overlay.add_trace(go.Box(y=b[y], name="Before", boxpoints="outliers"))
        overlay.add_trace(go.Box(y=a[y], name="After", boxpoints="outliers"))
        overlay.update_layout(title=f"Overlay: {y} (Before vs After)", yaxis_title=y)
        return fig_b, fig_a, overlay

    if t == "count":
        x = cand["x"]
        vc_b = df_before[x].value_counts(dropna=False).head(30)
        vc_a = df_after[x].value_counts(dropna=False).head(30)


        plot_b = vc_b.reset_index()
        plot_b.columns = [x, "count"]
        plot_b[x] = plot_b[x].astype(str)

        plot_a = vc_a.reset_index()
        plot_a.columns = [x, "count"]
        plot_a[x] = plot_a[x].astype(str)


        fig_b = px.bar(
            x=x, 
            y="count",
            title=f"Before: Top values of {x}"
        )

        fig_a = px.bar(
            x=x,
            y="count",
            title=f"After: Top values of {x}"
        )

        fig_b.update_layout(xaxis_tickangle=-45)
        fig_a.update_layout(xaxis_tickangle=-45)
        return fig_b, fig_a, None

    if t == "scatter":
        x, y = cand["x"], cand["y"]
        fig_b = px.scatter(b, x=x, y=y, opacity=0.7, title=f"Before: {x} vs {y}")
        fig_a = px.scatter(a, x=x, y=y, opacity=0.7, title=f"After: {x} vs {y}")

        # Overlay scatter (two traces)
        overlay = go.Figure()
        overlay.add_trace(go.Scattergl(x=b[x], y=b[y], mode="markers", name="Before", opacity=0.5))
        overlay.add_trace(go.Scattergl(x=a[x], y=a[y], mode="markers", name="After", opacity=0.5))
        overlay.update_layout(title=f"Overlay: {x} vs {y}", xaxis_title=x, yaxis_title=y)
        return fig_b, fig_a, overlay

    if t == "timeseries":
        x, y = cand["x"], cand["y"]
        bb = df_before.copy()
        aa = df_after.copy()
        bb[x] = pd.to_datetime(bb[x], errors="coerce")
        aa[x] = pd.to_datetime(aa[x], errors="coerce")
        bb = bb.dropna(subset=[x]).sort_values(x)
        aa = aa.dropna(subset=[x]).sort_values(x)

        fig_b = px.line(bb, x=x, y=y, title=f"Before: {y} over {x}")
        fig_a = px.line(aa, x=x, y=y, title=f"After: {y} over {x}")

        overlay = go.Figure()
        overlay.add_trace(go.Scatter(x=bb[x], y=bb[y], mode="lines", name="Before"))
        overlay.add_trace(go.Scatter(x=aa[x], y=aa[y], mode="lines", name="After"))
        overlay.update_layout(title=f"Overlay: {y} over {x}", xaxis_title=x, yaxis_title=y)
        return fig_b, fig_a, overlay

    return None, None, None

def handle_missing_values(state: DataState) -> DataState:
    """Fill missing numeric values with the column mean."""
    df = state["df"].copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())
    state["df"] = df
    return state


def remove_outliers(state: DataState) -> DataState:
    """Remove outliers using IQR method."""
    df = state["df"].copy()
    numeric_cols = df.select_dtypes(include="number").columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    state["df"] = df
    return state

def both(state: DataState) -> DataState:
    """Clean missing values and remove outliers."""
    df = state["df"].copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())
    
    """Remove outliers using IQR method."""
    numeric_cols = df.select_dtypes(include="number").columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    state["df"] = df
    
    return state

def lowercase_text_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    df2 = df.copy()

    if columns is None:
        columns = df2.select_dtypes(include="object").columns.tolist()

    for col in columns:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.lower()

    return df2

def tokenize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df2 = df.copy()

    for col in columns:
        if col in df2.columns:
            df2[col + "_tokens"] = (
                df2[col]
                .astype(str)
                .str.lower()
                .apply(lambda x: re.findall(r"\b\w+\b", x))
            )
            df2[col + "_token_count"] = df2[col + "_tokens"].apply(len)

    return df2

def describe_data(state: DataState) -> DataState:
    """Describe numeric columns after any cleaning."""
    state["summary"] = state["df"].describe().to_string()
    return state


def output_results(state: DataState):
    print(f"\n=== ACTION DECIDED: {state['action'].upper()} ===\n")
    print(state["summary"])

def branch_clean_missing(state: DataState) -> DataState:
    return state

def branch_remove_outliers(state: DataState) -> DataState:
    return state

def branch_both(state: DataState) -> DataState:
    return state

def branch_none(state: DataState) -> DataState:
    return state



# ---------------------------
# 4. Router Function
# ---------------------------

def route_action(state: DataState) -> str:
    """Route based on LLM's chosen action."""
    mapping = {
        "clean_missing": "branch_clean_missing",
        "remove_outliers": "branch_remove_outliers",
        "both": "branch_both",
        "none": "branch_none",
    }
    return mapping.get(state.get("action", "none"), "branch_none")


# ---------------------------
# 5. Build Graph
# ---------------------------
workflow = StateGraph(DataState)

workflow.add_node("load_data", load_data)
workflow.add_node("summarize_before", summarize_before)
workflow.add_node("reasoning_node", reasoning_node)

# branch label nodes
workflow.add_node("branch_clean_missing", branch_clean_missing)
workflow.add_node("branch_remove_outliers", branch_remove_outliers)
workflow.add_node("branch_both", branch_both)
workflow.add_node("branch_none", branch_none)

# real work nodes
workflow.add_node("apply_cleaning", apply_cleaning)
workflow.add_node("apply_text_processing", apply_text_processing)
workflow.add_node("summarize_after", summarize_after)
workflow.add_node("suggest_visualizations", suggest_visualizations)
workflow.add_node("output_state", output_state)

workflow.add_edge(START, "load_data")
workflow.add_edge("load_data", "summarize_before")
workflow.add_edge("summarize_before", "reasoning_node")

# conditional edges go to the label nodes (so the graph shows paths)
workflow.add_conditional_edges(
    "reasoning_node",
    route_action,
    {
        "branch_clean_missing": "branch_clean_missing",
        "branch_remove_outliers": "branch_remove_outliers",
        "branch_both": "branch_both",
        "branch_none": "branch_none",
    }
)

# converge all branches into the single cleaning node
workflow.add_edge("branch_clean_missing", "apply_cleaning")
workflow.add_edge("branch_remove_outliers", "apply_cleaning")
workflow.add_edge("branch_both", "apply_cleaning")
workflow.add_edge("branch_none", "output_state")

# continue pipeline
workflow.add_edge("apply_cleaning", "apply_text_processing")
workflow.add_edge("apply_text_processing", "summarize_after")
workflow.add_edge("summarize_after", "suggest_visualizations")
workflow.add_edge("suggest_visualizations", "output_state")
workflow.add_edge("output_state", END)

graph = workflow.compile()

# ---------------------------
# 6. Visualize Graph
# ---------------------------

def save_graph_visualization():
    """Save the workflow graph as a PNG image."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        output_path = PROJECT_ROOT / "outputs" / "workflow_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Workflow graph saved to outputs/workflow_graph.png\n")
    except Exception as e:
        print(f"Could not generate graph visualization: {e}\n")


# ---------------------------
# 7.0. Helpers
# ---------------------------
def make_summary(df: pd.DataFrame) -> dict:
    """Small, Streamlit-friendly summary blocks."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_by_col": df.isna().sum().sort_values(ascending=False),
        "missing_total": int(df.isna().sum().sum()),
        "numeric_describe": df[numeric_cols].describe().T if numeric_cols else pd.DataFrame(),
    }
    return summary

def detect_outliers_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outlier *cells* (not rows). Returns:
      - outlier_mask: boolean df aligned to numeric subset columns (same index)
      - outlier_table: tidy table of (row_index, column, value, lower, upper)
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        empty_mask = pd.DataFrame(False, index=df.index, columns=[])
        empty_tbl = pd.DataFrame(columns=["row_index", "column", "value", "lower_bound", "upper_bound"])
        return empty_mask, empty_tbl

    mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    records = []

    for col in numeric_cols:
        s = df[col]
        # Skip if all NaN
        if s.dropna().empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        # If iqr == 0, outlier detection is meaningless; treat as none
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        col_mask = (s < lower) | (s > upper)
        mask.loc[col_mask, col] = True

        for idx, val in s[col_mask].items():
            records.append(
                {"row_index": idx, "column": col, "value": val, "lower_bound": lower, "upper_bound": upper}
            )

    outlier_table = pd.DataFrame.from_records(records)
    return mask, outlier_table

def winsorize_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Clip numeric columns to IQR bounds (keeps all rows/cols)."""
    df2 = df.copy()
    numeric_cols = df2.select_dtypes(include="number").columns

    for col in numeric_cols:
        s = df2[col]
        if s.dropna().empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df2[col] = s.clip(lower, upper)

    return df2

def fill_missing_means(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with mean; leave non-numeric missing as-is (or extend as needed)."""
    df2 = df.copy()
    for col in df2.select_dtypes(include="number").columns:
        df2[col] = df2[col].fillna(df2[col].mean())
    return df2




def make_plotly_figure(df: pd.DataFrame, candidate: dict):
    t = candidate["type"]

    # For big data, sample for scatter/hist speed
    df_plot = df
    if len(df_plot) > 200_000 and t in {"scatter", "hist"}:
        df_plot = df_plot.sample(50_000, random_state=42)

    if t == "bar_missing_by_col":
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(30)
        if miss.empty:
            # still show something
            miss = df.isna().sum().head(30)
        fig = px.bar(
            x=miss.index.astype(str),
            y=miss.values,
            labels={"x": "Column", "y": "Missing count"},
            title="Missing values by column (top 30)",
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    if t == "heatmap_corr":
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2:
            return None
        corr = num.corr(numeric_only=True)
        fig = px.imshow(corr, title="Correlation heatmap (numeric columns)")
        return fig

    if t == "hist":
        x = candidate["x"]
        fig = px.histogram(df_plot, x=x, nbins=40, title=f"Distribution: {x}")
        return fig

    if t == "box":
        y = candidate["y"]
        fig = px.box(df_plot, y=y, points="outliers", title=f"Box plot: {y}")
        return fig

    if t == "count":
        x = candidate["x"]
        # top categories only
        vc = df[x].value_counts(dropna=False).head(30)
        fig = px.bar(
            x=vc.index.astype(str),
            y=vc.values,
            labels={"x": x, "y": "Count"},
            title=f"Top values: {x}",
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    if t == "scatter":
        x, y = candidate["x"], candidate["y"]
        fig = px.scatter(df_plot, x=x, y=y, title=f"Scatter: {x} vs {y}", opacity=0.7)
        return fig

    if t == "timeseries":
        x, y = candidate["x"], candidate["y"]
        dff = df.copy()
        dff[x] = pd.to_datetime(dff[x], errors="coerce")
        dff = dff.dropna(subset=[x])
        fig = px.line(dff.sort_values(x), x=x, y=y, title=f"Time series: {y} over {x}")
        return fig

    if t == "missingness_light":
        cols = candidate.get("cols", [])
        # show percent missing for specific cols
        if not cols:
            return None
        pct = (df[cols].isna().mean() * 100).sort_values(ascending=False)
        fig = px.bar(
            x=pct.index.astype(str),
            y=pct.values,
            labels={"x": "Column", "y": "% missing"},
            title="Missingness (% of rows) for selected columns",
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    return None



# ---------------------------
# 8.0 Streamlit UI
# ---------------------------
def run_workflow(file):
    # Save workflow visualization
    save_graph_visualization()

    # Dataframes
    df_before: pd.DataFrame
    df_after: pd.DataFrame

    # Audit tables
    missing_rows_before: pd.DataFrame           # rows that had missing values originally
    outlier_cells_before: pd.DataFrame          # row/col/value for detected outliers (before)

    # Summaries
    summary_before: dict
    summary_after: dict

    # Decision
    action: Literal["clean_missing", "remove_outliers", "both", "none"]

    return graph.invoke(init_state)

def handle_file_upload():
    if "workflow_result" not in st.session_state:
        st.session_state["workflow_result"] = None
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to see the workflow and run cleaning.")
        return

    # Save upload to temp path for pandas + your workflow
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getbuffer())
        temp_path = tmp.name

    st.success(f"Uploaded: {uploaded.name}")

    # Preview read ONLY for UI config (text columns, etc.)
    try:
        df_preview = pd.read_csv(temp_path)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    # Show workflow graph AFTER upload (no need to save to disk)
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        st.image(png_bytes, caption="Workflow graph")
    except Exception as e:
        st.warning(f"Could not render workflow graph: {e}")

    # Detect text columns safely
    text_cols = df_preview.select_dtypes(include="object").columns.tolist()

    # Put controls in a form so the button doesn't fight widget state
    with st.form("run_workflow_form"):
        st.subheader("Text processing options")

        if not text_cols:
            st.warning("No text columns detected. Text processing is disabled for this file.")
            text_option = "none"
            selected_text_cols = []
        else:
            selected_text_cols = st.multiselect(
                "Select text columns for processing",
                options=text_cols,
                default=[],
                key="selected_text_cols",
            )
            text_option = st.selectbox(
                "Text processing option",
                ["none", "lowercase", "tokenize", "both"],
                index=0,
            )

        run = st.form_submit_button("Run Cleaning Workflow")

    if not run:
        return

    # Run the workflow
    if run:
        init_state: DataState = {
            "csv_path": temp_path,
            "action": "none",
            "text_action": text_option,
            "text_columns": selected_text_cols,
        }
        with st.spinner("Running workflow..."):
            st.session_state["workflow_result"] = graph.invoke(init_state)


    result = st.session_state["workflow_result"]
    if result is None:
        st.info("Configure options above, then click Run Cleaning Workflow.")
        return

    # ---- RESULTS UI ----
    df_before = result.get("df_before")
    df_after = result.get("df_after", df_before)  # fallback

    if df_before is None:
        st.error("Workflow result missing df_before. Check load_data wiring.")
        return

    if df_after is None:
        st.error("Workflow result missing df_after and df_before. Check apply_cleaning.")
        return

    action = result.get("action", "none")
    st.subheader("Workflow decision")
    st.info(f"Selected option: **{action}**")

    label_map = {
        "clean_missing": "Clean missing values (fill numeric NaNs with mean)",
        "remove_outliers": "Remove outliers (winsorize/clip numeric outliers via IQR)",
        "both": "Both (winsorize outliers + fill missing values)",
        "none": "None (no cleaning applied)",
    }
    st.caption(label_map.get(action, action))

    # --- Side-by-side summaries ---
    st.markdown("### Summary (Before vs After)")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Before")
        st.write(f"Shape: {result['summary_before']['shape']}")
        st.write(f"Total missing: {result['summary_before']['missing_total']}")
        st.markdown("**Missing by column (top 15)**")
        st.dataframe(result["summary_before"]["missing_by_col"].head(15))
        st.markdown("**Numeric describe (first 15 numeric cols)**")
        st.dataframe(result["summary_before"]["numeric_describe"].head(15))

    with right:
        st.markdown("#### After")
        st.write(f"Shape: {result['summary_after']['shape']}")
        st.write(f"Total missing: {result['summary_after']['missing_total']}")
        st.markdown("**Missing by column (top 15)**")
        st.dataframe(result["summary_after"]["missing_by_col"].head(15))
        st.markdown("**Numeric describe (first 15 numeric cols)**")
        st.dataframe(result["summary_after"]["numeric_describe"].head(15))

    # --- Details: outliers + missing rows ---
    st.markdown("### What changed / what was flagged")
    out_tbl = result["outlier_cells_before"]
    miss_rows = result["missing_rows_before"]

    a, b = st.columns(2)

    with a:
        st.markdown("#### Outlier values detected (before)")
        if out_tbl.empty:
            st.info("No numeric outliers detected (IQR).")
        else:
            st.dataframe(out_tbl)

    with b:
        st.markdown("#### Rows that had missing values (before)")
        if miss_rows.empty:
            st.info("No missing values found.")
        else:
            st.dataframe(miss_rows)

    # --- Preview dataframes ---
    with st.expander("Preview dataframes"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before (head)**")
            st.dataframe(df_before.head(50))
        with c2:
            st.markdown("**After (head)**")
            st.dataframe(df_after.head(50))

    # --- Download cleaned CSV ---
    cleaned_csv_bytes = df_after.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download cleaned CSV",
        data=cleaned_csv_bytes,
        file_name=f"cleaned_{uploaded.name}",
        mime="text/csv",
    )

    # --- Visualizations ---
    st.markdown("### Recommended visualizations (Before vs After)")
    st.caption(result.get("viz_reason", ""))

    candidates = build_viz_candidates(df_before)
    cand_map = {c["id"]: c for c in candidates}
    selected_ids = result.get("viz_plan", {}).get("selected", [])

    for viz_id in selected_ids:
        cand = cand_map.get(viz_id)
        if not cand:
            continue

        fig_b, fig_a, fig_overlay = make_paired_figures(df_before, df_after, cand)
        if fig_b is None and fig_a is None:
            continue

        st.markdown(f"#### {viz_id}")
        col1, col2 = st.columns(2)
        with col1:
            if fig_b is not None:
                st.plotly_chart(fig_b, use_container_width=True)
        with col2:
            if fig_a is not None:
                st.plotly_chart(fig_a, use_container_width=True)

        if fig_overlay is not None:
            with st.expander("Overlay (Before vs After)"):
                st.plotly_chart(fig_overlay, use_container_width=True)
