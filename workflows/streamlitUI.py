import streamlit as st
import pandas as pd
import numpy as np
import simple_clean_data_workflow

st.set_page_config(layout="wide")
st.title("Data Cleaning Workflow")
st.text("This is a simple data cleaning agent that decides whether to clean missing values, remove outliers, or both.")
st.text("Users have the option of tokenization or lowercasing string columns. I didn't leave this option to the LLM's discretion since requirements can be domain specific.")
st.text("It also recommends visualizations to help understand the data.")
simple_clean_data_workflow.handle_file_upload()

