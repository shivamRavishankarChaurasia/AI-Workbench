import os
import time 
import glob

import numpy as np
import pandas as pd
import streamlit as st
import constants as c
import matplotlib.pyplot as plt
import seaborn as sns
import Utilities.py_tools as Manager
from datetime import datetime, timedelta

# Streamlit Page Config
st.set_page_config(layout="wide",page_title="Storage",page_icon="https://storage.googleapis.com/ai-workbench/Storage.svg")
Manager.faclon_logo()

st.subheader('Exploratory Data Analysis')
file_name = Manager.files_details()


if len(file_name)>1:
        if 'default_name' not in st.session_state:
            st.session_state.default_name = file_name[0]
        select_analysis = st.radio("Yooo",["Univariant Analysis" , "Bivariant Analysis" , "Multivariant Analysis"],horizontal=True,label_visibility="collapsed")
        st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)

        if select_analysis == "Univariant Analysis":
            st.subheader("Univariant Analysis", help="This section provides insights of the single column through visualization")
            col1,col2 = st.columns([3.5,1.5])
            tab0_name = col2.selectbox("Please select file:",file_name,key="tab0_name",index=file_name.index(st.session_state['default_name']))
            st.session_state['default_name'] = tab0_name
            df = Manager.read_parquet(file_name=tab0_name)
            # st.write(df)
            type_of_data = col2.radio("Select the type of data" , ["Numerical" , "Categorical"] ,label_visibility = "collapsed" , horizontal = True)
            if type_of_data == "Numerical":
                selected_cols = col2.selectbox("Select the columns" , df.select_dtypes(include=np.number).columns.tolist())
                plots =col2.selectbox("Select the plots" ["Histogram" ,"Distplot" ,"Boxplot"])
                Manager.plot_numerical_data(df, selected_cols , plots , col1)
            else:
                selected_cols = col2.selectbox("Select the columns" , df.select_dtypes(include='object').columns.tolist())
                plots = col2.selectbox("Select the plots" , ["Count_plot" , "Pie chart"])
                Manager.plot_categorical_data(df , selected_cols , plots , col1)
            col2.button("Save the plot"  , type = 'primary' , use_container_width  = True)

             
