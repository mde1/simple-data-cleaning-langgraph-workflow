import streamlit as st
import pandas as pd
import numpy as np
import simple_clean_data_workflow

st.title("Data Cleaning Workflow")
st.image("outputs/workflow_graph.png", use_column_width=True)
st.button("Upload Data")
simple_clean_data_workflow.handle_file_upload()

