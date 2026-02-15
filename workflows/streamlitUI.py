import streamlit as st
import pandas as pd
import numpy as np
import simple_clean_data_workflow

st.set_page_config(layout="wide")
st.title("Data Cleaning Workflow")
st.image("outputs/workflow_graph.png")
simple_clean_data_workflow.handle_file_upload()

