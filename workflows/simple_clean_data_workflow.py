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

class DataState(TypedDict):
    csv_path: str
    df: pd.DataFrame
    action: Literal["clean_missing", "remove_outliers", "both"]
    summary: str


# ---------------------------
# 2. Initialize LLM
# ---------------------------

llm = ChatOpenAI(model="gpt-4o-mini")


# ---------------------------
# 3. Nodes
# ---------------------------

def load_data(state: DataState) -> DataState:
    """Load CSV into DataFrame."""
    state["df"] = pd.read_csv(state["csv_path"])
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


def reasoning_node(state: DataState) -> DataState:
    """Use LLM to decide whether to clean missing values or remove outliers."""
    prompt = (
        "You are a data science assistant. "
        "Given this dataset summary, decide which single action is most appropriate: "
        "'clean_missing', 'remove_outliers', or 'both'.\n\n"
        "Both means clean missing values and remove outliers. If possible, choose 'both' if it makes sense."
        "Don't drop any rows or columns."
        f"{state['summary']}\n\n"
        "Respond only with one of: clean_missing, remove_outliers, both."
    )
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
workflow.add_node("summarize_data", summarize_data)
workflow.add_node("reasoning_node", reasoning_node)
workflow.add_node("handle_missing_values", handle_missing_values)
workflow.add_node("remove_outliers", remove_outliers)
workflow.add_node("describe_data", describe_data)
workflow.add_node("output_results", output_results)
workflow.add_node("both", both)

workflow.add_edge(START, "load_data")
workflow.add_edge("load_data", "summarize_data")
workflow.add_edge("summarize_data", "reasoning_node")
workflow.add_conditional_edges("reasoning_node", route_action, {
    "handle_missing_values": "handle_missing_values",
    "remove_outliers": "remove_outliers",
    "describe_data": "describe_data",
    "both": "both"
})
workflow.add_edge("handle_missing_values", "describe_data")
workflow.add_edge("remove_outliers", "describe_data")
workflow.add_edge("describe_data", "output_results")
workflow.add_edge("both", "describe_data")
workflow.add_edge("output_results", END)

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
# 6.1. Streamlit UI
# ---------------------------
def run_workflow():
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

    return graph.invoke(init_state)

def handle_file_upload():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Extra safety check (even though type=["csv"] filters it)
        if not uploaded_file.name.lower().endswith(".csv"):
            st.error("Please upload a valid CSV file.")
            return

        # Save to a temporary file so your workflow can access it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        st.success("File uploaded successfully.")

        if st.button("Run Workflow"):
            with st.spinner("Processing..."):
                result = run_workflow(temp_path)

            st.success("Workflow complete.")
            st.write(result)

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