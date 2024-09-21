import os
import time 
import glob

import numpy as np
import pandas as pd
import streamlit as st
import constants as c
import iosense_connect as io

import Utilities.py_tools as Manager
from datetime import datetime, timedelta

# Streamlit Page Config
st.set_page_config(layout="wide",page_title="Dashborad",page_icon="https://storage.googleapis.com/ai-workbench/Storage.svg")
Manager.faclon_logo()

st.subheader('Dashboard')
file_name = Manager.files_details()
st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)

if len(file_name)>0:
    col1, col2 = st.columns([2,2])
    with col1:
     st.write("Col1")
    with col2:
     st.write("Col2")


    


