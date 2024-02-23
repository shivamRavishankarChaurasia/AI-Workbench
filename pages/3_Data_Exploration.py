import os
import glob

import constants as c
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import Utilities.py_tools as Manager
from api import generate_dashboard
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(layout="wide",page_title="Data Exploration",page_icon="https://storage.googleapis.com/ai-workbench/Data%20Exploration.svg")
Manager.faclon_logo()

file_name = Manager.files_details()
st.subheader('Data Exploration')

if len(file_name) > 0:

    if 'default_name' not in st.session_state:
            st.session_state.default_name = file_name[0]
    
    tab = st.radio('Select tabs', ['One-Click Plot','Manual Plot', 'Time Series', 'Plots', '3D Plots'], horizontal=True,  index=0, key='radio_key', label_visibility='collapsed')
    st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)

    if tab == 'One-Click Plot':
        # One-Click plot contains Correlation heatmap & Pairplot
        col1,col2 = st.columns([3.5,1.5])
        container = col1.empty() # container that will hold all plots

        tab1_file_name = col2.selectbox("Please Select file:",file_name,key="tab1_file_name", index=file_name.index(st.session_state['default_name']))
        st.session_state.default_name = tab1_file_name
        tab1_df = Manager.read_parquet(file_name=st.session_state.tab1_file_name)
        #col1.write(tab1_df)
        # Dropping date column(not required)
        date_column = tab1_df.select_dtypes(include=['datetime']).columns.to_list()
        tab1_df.drop(columns=date_column, axis=1, inplace=True)
        
        # Heatmap will be drawn btw all numeric columns & selected categorical columns(encoded to numeric)
        # Ensuring df have atleast 2 columns
        if len(tab1_df.columns.to_list())<2:
            container.error('Not enough columns to draw plot')
        
        else:
            tab1_select = col2.radio("Please Select Type:",['Correlation','Pairplot'],horizontal=True)
            if tab1_select == 'Correlation':
                #keeping only categorical columns and dropping remaining object dtype cols
                categorical_columns = Manager.determine_categorial_columns(tab1_df,threshold=0.03)
                obj_columns = tab1_df.select_dtypes(include=['object']).columns
                remaining_col = list(set(obj_columns) - set(categorical_columns))

                #dropped those object columns not classified as categorical col
                tab1_df.drop(remaining_col,axis=1,inplace=True) 

                # Checkbox for selecting categorical columns
                random_df = pd.DataFrame(categorical_columns,columns=['Categorical'])
                random_df['Checkbox'] = False   
        
                col2.info("Select Categorical Columns")
                check_df = col2.data_editor(random_df,height=200,use_container_width=True,hide_index=True,
                                    column_config={
                    "Checkbox": st.column_config.CheckboxColumn(
                        "Checkbox",
                        default=False
                    )})

                if True in check_df['Checkbox'].values:
                    # Heatmap for numeric & selected categorical columns
                    selected_columns = check_df.loc[check_df['Checkbox'], 'Categorical'].tolist()
                    try:
                        tab1_df = pd.get_dummies(tab1_df, columns=selected_columns)
                    except MemoryError as e:
                        st.error('High Memory Usage..Cant perform operation')


                    non_selected = [col for col in categorical_columns if col not in selected_columns]
                    tab1_df.drop(non_selected, axis=1, inplace=True) #dropped non-selected categorical columns in checkbox
                else:
                    # Heatmap for numeric columns only
                    tab1_df.drop(categorical_columns,axis=1, inplace=True) #dropped all categorical columns
                    
                correlation_df = tab1_df.corr()
                        
                if not correlation_df.empty:  # Check if correlation_df is not empty
                    try:
                        fig = Manager.corr_plot(data = correlation_df)
                        container.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        container.error(f"Reason: {e}")
                else:
                    container.warning("No numeric columns available for correlation plot. Please select categorical columns")
            
            # Pairplot will include plots of selected columns only from Checkbox
            if tab1_select == 'Pairplot':
                # Dropping object columns(as Pairplot is drawn btw numeric col only)
                obj_columns = tab1_df.select_dtypes(include=['object']).columns
                tab1_df.drop(columns=obj_columns, axis=1, inplace=True)

                if tab1_df.empty:
                    container.warning('Dataset has no Numeric columns to plot.')
                else:
                    # Checkbox for selecting col
                    col2_df = pd.DataFrame(tab1_df.columns,columns=['Columns'])
                    col2_df['Checkbox'] = False
                    col2_df['Checkbox'].iloc[0] = True
                    col2.info("Select Dataframe Columns")
                    check_df = col2.data_editor(col2_df,height=200,use_container_width=True,hide_index=True,column_config={"Checkbox": st.column_config.CheckboxColumn("Checkbox",default=False)})
            
                    if True in check_df['Checkbox'].values:
                        selected_columns = check_df.loc[check_df['Checkbox'], 'Columns'].to_list()
                
                    try:
                        fig= Manager.pairplot_fig(data=tab1_df, col=selected_columns)
                        container.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        container.error(f"Reason: {e}")
        if col2.button("Save the plot", type = "primary" , use_container_width=True):
            generate_dashboard(tab1_file_name,fig)
            col2.success("Plots Saved Sucessfully")

    if tab == 'Manual Plot':
        col1,col2 = st.columns([3.5,1.5])
        tab2_file_name = col2.selectbox("Please Select file:",file_name,key="tab2_file_name", index=file_name.index(st.session_state.default_name))
        st.session_state.default_name = tab2_file_name
    
        tab2_df = Manager.read_parquet(file_name=tab2_file_name)
        container = col1.empty()

        if len(tab2_df.columns.to_list()) >= 2: #Ensuring min 2 col
            tab2_select = col2.selectbox("Please Select Type:",['Line Chart', 'Bar Chart', 'Scatter Plot', 'Area Chart', 'Histogram', 'Pie Chart'])
            #col1.write(tab2_df)
        
            if tab2_select == 'Line Chart':
                line_col1, line_col2 = col2.columns(2)
                x_col = line_col1.selectbox('Select x_axes:', tab2_df.columns.to_list(), key='x_col')

                # remaining_col will include all columns except x_col
                remaining_col = [col for col in tab2_df.columns.to_list() if col != x_col]
                # Multiple select option for y column 
                y_col = line_col2.multiselect('Select y_axes:', remaining_col, key='y_col')

                if len(y_col) ==0: #if no y col is selected
                    container.info("Please select y axis for the graph")

                else:
                    fig = Manager.create_plotly_chart('line',tab2_df, x_col, y_col)
                    container.plotly_chart(fig, use_container_width=True)
                    
            if tab2_select == 'Bar Chart':
                bar_col1, bar_col2 = col2.columns(2)
                x_col = bar_col1.selectbox('Select x_axes:', tab2_df.columns.to_list(), key='x_col')

                remaining_col = [col for col in tab2_df.columns.to_list() if col != x_col]
                y_col = bar_col2.multiselect('Select y_axes:', remaining_col, key='y_col')

                if len(y_col) ==0:
                    container.info("Please select y axis for the graph")
                else:
                    fig = Manager.create_plotly_chart('bar',tab2_df, x_col, y_col)
                    container.plotly_chart(fig, use_container_width=True)

            if tab2_select == 'Scatter Plot':
                sct_col1, sct_col2 = col2.columns(2)
                x_col = sct_col1.selectbox('Select x_axes:', tab2_df.columns.to_list(), key='x_col')

                remaining_col = [col for col in tab2_df.columns.to_list() if col != x_col]
                y_col = sct_col2.multiselect('Select y_axes:', remaining_col, key='y_col')
           
                if len(y_col) ==0:
                    container.info("Please select y axis for the graph")
                              
                else:
                    use_color = col2.checkbox('Color markers by column?', key='use_color') # Marker color by column  

                    if use_color:
                        # dropping date column as not required in color argument for scatter plot
                        date_column = tab2_df.select_dtypes(include=['datetime']).columns.to_list()
                        remaining_col_color = [col for col in tab2_df.columns.to_list() if (col!= x_col and col!= y_col and col not in date_column)]
                        c_col = col2.selectbox('Select color by:', remaining_col_color, key='c_col')
                    else:
                        c_col = None

                    # slider for marker size
                    size= col2.slider('Select marker size', min_value=1, max_value=10, value=5,step=1, key='sctsize')

                    try:
                        fig = Manager.scatter_plot(tab2_df, x_col, y_col, c_col, size)
                        container.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        container.error(f"Reason: {e}")


            if tab2_select == 'Area Chart':
                area_col1, area_col2 = col2.columns(2)
                x_col = area_col1.selectbox('Select x_axes:', tab2_df.columns.to_list(), key='x_col')

                remaining_col = [col for col in tab2_df.columns.to_list() if col != x_col]
                y_col = area_col2.multiselect('Select y_axes:', remaining_col, key='y_col')
 
                if len(y_col) == 0:
                    container.info("Please select y axis for the graph")
                else:
                    fig = Manager.create_plotly_chart('area',tab2_df, x_col, y_col)
                    container.plotly_chart(fig, use_container_width=True)

            if tab2_select == 'Histogram':
                hist_col1, hist_col2 = col2.columns(2)
                x_col = hist_col1.selectbox('Select x_axes:', tab2_df.columns.to_list(), key='x_col')

                remaining_col = [col for col in tab2_df.columns.to_list() if col != x_col]
                y_col = hist_col2.multiselect('Select y_axes:', remaining_col, key='y_col')

                if len(y_col) ==0:
                    container.info("Please select y axis for the graph")

                else:
                    fig = Manager.create_plotly_chart('histogram',tab2_df, x_col, y_col)
                    container.plotly_chart(fig, use_container_width=True)
            
            # Pie chart will show perc distribution(value i.e. numeric col) for respective categorical feature(name)
            if tab2_select == 'Pie Chart':
                pie_col1, pie_col2 = col2.columns(2)
                
                # dropping date column(not required for pie chart)
                date_column = tab2_df.select_dtypes(include=['datetime']).columns.to_list()
                tab2_df.drop(columns=date_column, axis=1, inplace=True)
                
                obj_columns = tab2_df.select_dtypes(include=['object']).columns
                name = pie_col1.selectbox('Select Keys:', obj_columns, key='name', help= 'Select categorical columns')

                remaining_col = [col for col in tab2_df.columns.to_list() if col not in obj_columns]
                value = pie_col2.selectbox('Select Data:', remaining_col, key='value', help='Select columns with numeric values')

                if name is None:
                    container.error('No categorical columns in dataframe')
                else:
                    try:
                        fig = Manager.pie_chart(tab2_df,value, name )
                        container.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        container.error(f"Reason: {e}")
        if col2.button("Save the plot", type = "primary" , use_container_width=True):
            generate_dashboard(tab1_file_name ,fig)
            col2.success("Plots Saved Sucessfully")
        else:
            container.error('Not enough columns to draw plot')    

            

    if tab == 'Time Series':
        # For this tab, dataframe must contain a datetime column for Time-Series analysis
        col1,col2 = st.columns([3.5,1.5])
        tab3_file_name = col2.selectbox("Please Select file:",file_name,key="tab3_file_name", index=file_name.index(st.session_state.default_name))
        st.session_state.default_name = tab3_file_name

        # Dropping object columns   
        tab3_df = Manager.read_parquet(file_name=tab3_file_name)
        obj_columns = tab3_df.select_dtypes(include=['object']).columns
        tab3_df.drop(obj_columns,axis=1, inplace=True)

        date_column = tab3_df.select_dtypes(include=['datetime']).columns.to_list() 
        
        if date_column:
            # setting date column as df index
            tab3_df = tab3_df.set_index(date_column)

            if len(tab3_df.columns.to_list())>0: # Ensuring min 1 numeric col 
                tab3_df.dropna(inplace=True)

                tab3_select = col2.selectbox("Please Select Type:",['Time-Series Graph','Time-Series Decomposition', 'ACF', 'PACF'])
                y_col = col2.selectbox('Select Column:', tab3_df.columns.to_list(), key='ts_y_col')
            
                if tab3_select == 'Time-Series Graph':
                    container = col1.empty()
                    color=col2.color_picker('Select Color',"#007AF9", key='tsg_color')
                    fig = Manager.time_series_plot(tab3_df, y_col, color)
                    container.plotly_chart(fig, use_container_width=True)
            

                if tab3_select == 'Time-Series Decomposition':
                    col1.subheader("Time-series Decomposition",help="Decompose a time-series into its components: Trends, Seasonality and Residuals/Noise")
                    container = col1.empty()
                    type_select=col2.radio('Type:', ['Additive', 'Multiplicable'], horizontal=True )
                    model = 'additive' if type_select == 'Additive' else 'multiplicable'

                    try:
                        fig = Manager.decomposition_plot(tab3_df,y_col, model)
                        container.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        container.error(f"Reason: {e}")

                if tab3_select == 'ACF':
                    col1.subheader("Auto-Correlation Function",help=" correlation between a time series with a lagged version of itself")
                    container = col1.empty()
                    # slider for selecting no. of lags
                    acf_lags = col2.slider('Select no. of lags',min_value=10, max_value=50, value=20, step=5, key='acf')
                    
                    fig_acf = Manager.acf_plot(tab3_df, y_col, acf_lags)
                    container.pyplot(fig_acf, use_container_width=True)

                if tab3_select == 'PACF':
                    col1.subheader("Partial Auto-Correlation Function",help="partial correlation of a time series with its own lagged values")
                    container = col1.empty()
                    pacf_lags = col2.slider('Select no. of lags',min_value=10, max_value=50, value=20, step=5, key='pacf')
          
                    fig_pacf = Manager.pacf_plot(tab3_df, y_col, pacf_lags)
                    container.pyplot(fig_pacf, use_container_width=True)
            else:
                col1.error('Not enough columns to draw plot')
        if col2.button("Save the plot", type = "primary" , use_container_width=True):
            generate_dashboard(tab1_file_name ,fig)
            col2.success("Plots Saved Sucessfully")
        else:
            col1.error('No Datetime column present in dataframe')
    

    if tab == 'Plots':
        # This tab contains t-SNE & PCA plot
        col1,col2 = st.columns([3.5,1.5])

        tab4_file_name = col2.selectbox("Please Select file:",file_name,key="tab4_file_name",  index=file_name.index(st.session_state.default_name))
        st.session_state.default_name = tab4_file_name
        tab4_df = Manager.read_parquet(file_name=tab4_file_name)

        # Finding any datetime column & dropping it(since not required)
        date_column = tab4_df.select_dtypes(include=['datetime']).columns.to_list()
        tab4_df.drop(date_column,axis=1,inplace=True)
        # dropping null values
        tab4_df.dropna(inplace=True)

        # dropping object columns not classified as categorical columns
        categorical_columns = Manager.determine_categorial_columns(tab4_df,threshold=0.03)
        obj_columns = tab4_df.select_dtypes(include=['object']).columns.to_list()
        remaining_col = list(set(obj_columns) - set(categorical_columns))
        tab4_df.drop(remaining_col,axis=1,inplace=True)


        if len(tab4_df.columns.to_list()) >2:
            tab4_select = col2.radio("Please Select Type:",['PCA Plot','t-SNE Plot',],horizontal=True)

            
            # Checkbox for selecting categorical columns that will be included in plot
            random_df = pd.DataFrame(categorical_columns,columns=['Categorical'])
            random_df['Checkbox'] = False   
        
            col2.info("Select Categorical Columns")
            check_df = col2.data_editor(random_df,height=200,use_container_width=True,key='pca',hide_index=True,column_config={"Checkbox": st.column_config.CheckboxColumn("Checkbox",default=False)})
            
            if True in check_df['Checkbox'].values:
                selected_columns = check_df.loc[check_df['Checkbox'], 'Categorical'].tolist()
                tab4_df = pd.get_dummies(tab4_df, columns=selected_columns)
                # dropping non-selected columns
                non_selected = [col for col in categorical_columns if col not in selected_columns]
                tab4_df.drop(non_selected, axis=1, inplace=True)

            else:
                tab4_df.drop(categorical_columns, axis=1, inplace=True)# dropping all categorical columns

            # when df has only 1 non-categorical column, rest are all categorical cols
            if len(tab4_df.columns.to_list()) < 2:
                col1.warning('Not enough columns to draw plot. Please select categorical columns to proceed')
                
            else:
                # Scaling Values
                scaler = StandardScaler()
                scaler.fit(tab4_df)
                scaled_data = scaler.transform(tab4_df)

                size= col2.slider('Select marker size', min_value=1, max_value=10, value=5,step=1, key='size')
                color=col2.color_picker('Select Color', "#007AF9", key='color')

                
                if tab4_select == "PCA Plot":
                    col1.subheader("Principal Component Analysis", help=" reduces the dimensionality of a dataset and converts a set of correlated variables to a set of uncorrelated variables")
                    container = col1.empty()

                    pca = PCA(n_components=2)
                    pca.fit(scaled_data)
                    x_pca = pca.transform(scaled_data)
                    pca_df = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2'])

                    fig_pca = Manager.pca_plot(pca_df, size, color)
                    container.plotly_chart(fig_pca, use_container_width=True)

                if tab4_select == 't-SNE Plot':
                    col1.subheader("t-SNE", help='non-linear dimensionality reduction algorithm for exploring high-dimensional data.')
                    container = col1.empty()
                    m= TSNE(n_components=2)
                    tsne_features= m.fit_transform(scaled_data)
                    tsne_df = pd.DataFrame(data=tsne_features, columns=['X', 'Y'])

                    fig_tsne = Manager.tsne_plot(tsne_df, size, color)
                    container.plotly_chart(fig_tsne, use_container_width=True)
        if col2.button("Save the plot", type = "primary" , use_container_width=True):
            generate_dashboard(tab1_file_name ,fig)
            col2.success("Plots Saved Sucessfully")
        else:
            col1.error('Not enough columns to draw plots')
  
  
    if tab== '3D Plots':
        col1,col2 = st.columns([3.5,1.5])
        tab5_file_name = col2.selectbox("Please Select file:",file_name,key="tab5_file_name",  index=file_name.index(st.session_state.default_name))
        st.session_state.default_name = tab5_file_name
        tab5_df = Manager.read_parquet(file_name=tab5_file_name)
        container = col1.empty()
        
        #col1.write(tab5_df)
        if len(tab5_df.columns.to_list()) < 3:
            container.error('Not enough columns to draw 3D Graph')
        else:
            tab5_select = col2.radio("Please Select Type:",['Line Chart','Scatter Plot', 'Surface Plot'], horizontal = True)

            if tab5_select == 'Line Chart':
                line_col1, line_col2 = col2.columns(2)
                x_col = line_col1.selectbox('Select x_axes:', tab5_df.columns.to_list(), key='x_col_3d')

                remaining_col = [col for col in tab5_df.columns.to_list() if col != x_col]
                y_col = line_col2.selectbox('Select y_axes:', remaining_col, key='y_col_3d')
                remaining_col_z = [col for col in tab5_df.columns.to_list() if (col != x_col and col != y_col)]
                z_col = line_col1.selectbox('Select z_axes:', remaining_col_z, key='z_col_3d')

                try:
                    fig = Manager.line_3d_plot(tab5_df,x_col, y_col,z_col)
                    container.plotly_chart(fig, use_container_width=True)
            
                except Exception as e:
                    container.error(f"Reason: {e}")
            


            if tab5_select == 'Scatter Plot':
                sct_col1, sct_col2 = col2.columns(2)
                x_col = sct_col1.selectbox('Select x_axes:', tab5_df.columns.to_list(), key='x_col_3d')

                remaining_col = [col for col in tab5_df.columns.to_list() if col != x_col]
                y_col = sct_col2.selectbox('Select y_axes:', remaining_col, key='y_col_3d')
                remaining_col_z = [col for col in tab5_df.columns.to_list() if (col != x_col and col != y_col)]
                z_col = sct_col1.selectbox('Select z_axes:', remaining_col_z, key='z_col_3d')

                use_color = col2.checkbox('Color markers by column?',key='use_color_3d')
                
                if use_color:
                    date_column = tab5_df.select_dtypes(include=['datetime']).columns.to_list()
                    remaining_col_color = [col for col in tab5_df.columns.to_list() if (col != x_col and col != y_col and col!=z_col and col not in date_column)]
                    c_col = col2.selectbox('Select color by:', remaining_col_color, key='c_col_3d')
                else:
                    c_col = None

                size= col2.slider('Select marker size', min_value=1, max_value=10, value=5,step=1, key='sctsize_3d')

                try:
                    fig = Manager.scatter_3d_plot(tab5_df, x_col, y_col, z_col, c_col, size)
                    container.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    container.error(f"Reason: {e}")
        
                
            if tab5_select == 'Surface Plot':
                srf_col1, srf_col2 = col2.columns(2)

                x_col = srf_col1.selectbox('Select x_axes:', tab5_df.columns.to_list(), key='x_col_3d')

                remaining_col = [col for col in tab5_df.columns.to_list() if col != x_col]
                y_col = srf_col2.selectbox('Select y_axes:', remaining_col, key='y_col_3d')

                remaining_col_z = [col for col in tab5_df.columns.to_list() if (col != x_col and col != y_col)]
                z_col = srf_col1.selectbox('Select z_axes:', remaining_col_z, key='z_col_3d')

                try:
                    fig = go.Figure(data=[go.Surface(z=tab5_df[z_col], x=tab5_df[x_col], y=tab5_df[y_col])])
                    fig = Manager.surface_3d_plot(tab5_df, x_col, y_col, z_col)
                    container.plotly_chart(fig, use_container_width=True)
            
                except Exception as e:
                    container.error(f"Reason: {e}")
        if col2.button("Save the plot", type = "primary" , use_container_width=True):
            generate_dashboard(tab1_file_name ,fig)
            col2.success("Plots Saved Sucessfully")
else:
    st.error("Please import CSVs to perform operations")