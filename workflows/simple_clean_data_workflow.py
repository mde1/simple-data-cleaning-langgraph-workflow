from typing_extensions import TypedDict, Literal
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pathlib import Path
import io
import streamlit as st
import tempfile

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

def describe_data(state: DataState) -> DataState:
    """Describe numeric columns after any cleaning."""
    state["summary"] = state["df"].describe().to_string()
    return state


def output_results(state: DataState):
    print(f"\n=== ACTION DECIDED: {state['action'].upper()} ===\n")
    print(state["summary"])


# ---------------------------
# 4. Router Function
# ---------------------------

def route_action(state: DataState) -> str:
    """Route based on LLM's chosen action."""
    mapping = {
        "clean_missing": "handle_missing_values",
        "remove_outliers": "remove_outliers",
        "both": "both",
        "none": "describe_data",
    }
    return mapping.get(state["action"], "describe_data")


# ---------------------------
# 5. Build Graph
# ---------------------------

workflow = StateGraph(DataState)

workflow.add_node("load_data", load_data)
workflow.add_node("summarize_before", summarize_before)
workflow.add_node("reasoning_node", reasoning_node)
workflow.add_node("apply_cleaning", apply_cleaning)
workflow.add_node("summarize_after", summarize_after)
workflow.add_node("output_state", output_state)

workflow.add_edge(START, "load_data")
workflow.add_edge("load_data", "summarize_before")
workflow.add_edge("summarize_before", "reasoning_node")
workflow.add_edge("reasoning_node", "apply_cleaning")
workflow.add_edge("apply_cleaning", "summarize_after")
workflow.add_edge("summarize_after", "output_state")
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

    
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])

    if uploaded is not None:
        # Save upload to temp path for pandas + your workflow
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getbuffer())
            temp_path = tmp.name

        st.success(f"Uploaded: {uploaded.name}")

        if st.button("Run Cleaning Workflow"):
            with st.spinner("Running workflow..."):
                init_state: DataState = {"csv_path": temp_path, "action": "none"}
                result: DataState = graph.invoke(init_state)

            df_before = result["df_before"]
            df_after = result["df_after"]
            action = result.get("action", "none")

            st.subheader(f"Decision: `{action}`")

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

            action = result.get("action", "none")

            st.subheader("Workflow decision")
            st.info(f"Selected option: **{action}**")

            # Optional: friendlier labels
            label_map = {
                "clean_missing": "Clean missing values (fill numeric NaNs with mean)",
                "remove_outliers": "Remove outliers (winsorize/clip numeric outliers via IQR)",
                "both": "Both (winsorize outliers + fill missing values)",
                "none": "None (no cleaning applied)",
            }
            st.caption(label_map.get(action, action))

            a, b = st.columns(2)

            with a:
                st.markdown("#### Outlier values detected (before)")
                if out_tbl.empty:
                    st.info("No numeric outliers detected (IQR).")
                else:
                    st.caption("These are the original outlier cells. If action included outliers, values were clipped to IQR bounds.")
                    st.dataframe(out_tbl)

            with b:
                st.markdown("#### Rows that had missing values (before)")
                if miss_rows.empty:
                    st.info("No missing values found.")
                else:
                    st.caption("These rows had ≥1 missing value before cleaning (rows were not dropped).")
                    st.dataframe(miss_rows)

            # --- Show sample dataframes (optional) ---
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



# ---------------------------
# 7. Run Example
# ---------------------------

if __name__ == "__main__":
    # Save workflow visualization
    save_graph_visualization()
    
    # Run the workflow
    csv_path = str(PROJECT_ROOT / "data" / "missing_and_outliers.csv")
    init_state: DataState = {
        "csv_path": csv_path,
        "df": None,
        "action": "none",
        "summary": "",
    }
    graph.invoke(init_state)