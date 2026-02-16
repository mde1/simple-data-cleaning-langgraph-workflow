# LangGraph Workflow Example

A simple example demonstrating how to build workflows with LangGraph for data processing.

## How It Works
**Live Demo:** https://simple-data-cleaning-langgraph-workflow.onrender.com
The demo may take a minute to restart if it hasn't been accessed recently.

The workflow follows these steps:

1. **Load Data** - Reads the CSV file into a pandas DataFrame
2. **Summarize Data** - Generates comprehensive summary including:
   - Statistical description (`.describe()`)
   - Dataset info (`.info()`)
   - Explicit missing value counts
3. **LLM Reasoning** - Uses GPT-4o-mini to analyze the summary and decide which action to take
4. **Conditional Routing** - Routes to appropriate cleaning node based on LLM decision, LLM selects visualizations for the data type.
5. **Data Cleaning** - Can execute:
   - **Handle Missing Values** - Fills numeric missing values with column means
   - **Winsorize Outliers** - Detect and winsorize numeric outliers (IQR method)
6. **User Text Options** - Users can manually select columns to tokenize.
6. **Describe Data** - Generates statistical summary of cleaned data
7. **Output Results** - Prints the action taken and final summary
8. **Visualize the Data** - The AI agent selects the best visualizations for the data and displays before and after paired plots. 

## Setup

### Windows (PowerShell)

1. **Verify Python is installed** (Python 3.9 or higher required):
   ```powershell
   python --version
   ```
   If not installed, download from [python.org](https://www.python.org/downloads/)

2. **Install Poetry**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   After installation, restart your terminal or IDE. If `poetry` command is not found, add `%APPDATA%\Python\Scripts` to your system PATH.

3. **Install dependencies**:
   ```powershell
   poetry install
   ```

4. **Set up your OpenAI API key**:
   ```powershell
   copy .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`

### macOS/Linux

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Set up your OpenAI API key**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`

## Running the Example

From the project root:
```bash
poetry run python workflows/simple_clean_data_workflow.py
```

The workflow will:
1. Save a workflow graph visualization to `outputs/workflow_graph.png`
2. Load data from `data/missing.csv` (you can change this in the script)
3. Use an LLM to analyze the data and decide which cleaning action to take
4. Execute the appropriate cleaning steps
5. Display the results

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── data/
│   ├── missing.csv              # Data with only missing values
│   ├── outliers.csv             # Data with only outliers
│   └── missing_and_outliers.csv # Data with both issues
│
├── workflows/
│   └── simple_clean_data_workflow.py   # Main workflow implementation
│
└── outputs/
    └── .gitkeep                 # Generated files (graphs, reports)
```

**Folder Organization:**
- `data/` - Sample CSV files with different data quality issues
- `workflows/` - LangGraph workflow scripts
- `outputs/` - Generated outputs (visualizations, reports)
