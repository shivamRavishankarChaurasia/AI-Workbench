import os
import glob
import time 

import numpy as np
import constants as c
import streamlit as st
import pandas as pd
import Utilities.py_tools as Manager
from datetime import datetime, timedelta

st.set_page_config(layout="wide",page_title="Data Preprocessing",page_icon="https://img.icons8.com/?size=100&id=7O4JH5C7TOv5&format=png&color=000000")
Manager.faclon_logo()  

try:
    if "reset_disable" not in st.session_state:
         st.session_state.reset_disable = False

    file_name = Manager.files_details()
    st.subheader("Data Pre-Processing")

    if len(file_name) > 2:
        operation = st.radio(label="Selection Operation",  options=['Outlier Detection', 'Data Imbalance', 'Data Transformation'], horizontal=True, label_visibility="collapsed")
        st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)
        if  'default_name' not in  st.session_state or st.session_state.default_name not in file_name:
            st.session_state.default_name = file_name[0]
            
        if operation == "Outlier Detection":
            st.session_state.btn_rm_outlier = True
            st.subheader(operation) 
            ini_col1 ,ini_col2 = st.columns([3.5,1.2])
            col1, col2 = ini_col1.columns([3,2])
         # Create placeholders for charts and data
            placeholder1 = col1.empty()
            placeholder2 = col2.empty()
            placeholder3 = col2.empty()
        # widgets use to select the columns , dataframe , and methods         
            selected_dataset = ini_col2.selectbox("Please Select the file:", file_name , disabled = False ,
                                                   key="selected_dataset",index=file_name.index(st.session_state['default_name']))
            st.session_state['default_name'] =selected_dataset 
            data_frame = Manager.read_parquet(file_name = selected_dataset)
            selected_col = ini_col2.selectbox("Select the Column", data_frame.select_dtypes(include=np.number).columns.tolist())
            selected_method = ini_col2.radio("Type of Operation",("Inter Quantile Range" ,"Z-Score"), horizontal=True, 
                                             label_visibility="collapsed")
        # Check outliers and display them by using pie chart 
            df_outliers = Manager.get_outliers(data_frame , selected_col ,selected_method)
        # return data_frame df_outliers
            if len(df_outliers) >0:
                placeholder1.plotly_chart(Manager.plot_outlier_graphs(df_outliers , selected_col , selected_method),use_container_width = True)
                placeholder2.markdown(f"Outliers Detected in {selected_col}: {len(df_outliers)}")
                placeholder3.dataframe(df_outliers,hide_index=True
                                       ,height=350,use_container_width=True)
            else:
                ini_col1.warning("No outliers detected in the selected column", icon ="🔥")

            outliers_present = len(df_outliers) > 0

        # Now outliers are detected to remove  it click the Remove outliers button (internally it uses two method IQR and Z-score to remove the outliers ) 
            if ini_col2.button("Remove Outliers", use_container_width=True, type="primary", disabled= not outliers_present) or "outliers_remove" in st.session_state:
                if outliers_present:
                    st.session_state.outliers_remove = True
                remove_outliers =  Manager.drop_outliers(data_frame,df_outliers)
                placeholder1.plotly_chart(Manager.plot_outlier_graphs(remove_outliers , selected_col , selected_method),use_container_width = True)
                placeholder2.markdown(f"Dataframe without Outliers")
                placeholder3.dataframe(remove_outliers,hide_index=True , use_container_width = True )
        # Tabs for updating, saving, and downloading
                storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                with storage_tab1:
                    st.warning("Do you wish to overwrite the actual file")
                    if st.button("Update",key="random_update"):
                        try:
                            Manager.update_parquet(df=remove_outliers,file_name=selected_dataset)
                            Manager.modify_metadata(file_name=selected_dataset, new_process=[f"df = drop_outliers(df,get_outliers(df,'{selected_col}','{selected_method}'))"])
                            time.sleep(3)
                            Manager.delete_in_page_session()
                            st.experimental_rerun()
                        except Exception as e:
                            st.exception(e)
                with storage_tab2:
                    null_update = st.text_input("Please Provide Name",help="Please press enter to save the updated file \n Dont add .csv or any other extension",key="random_key")
                    if len(null_update) > 0 or 'null_update' in st.session_state:
                        st.session_state.null_update = True
                        Manager.create_parquet(df=remove_outliers,file_name=null_update)
                        Manager.delete_in_page_session()
                        time.sleep(3)
                        st.experimental_rerun()
                    else:
                        st.info('Please fill in inputs')
                with storage_tab3:
                    csv = remove_outliers.to_csv().encode('utf-8')
                    st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                    mime='text/csv',
                    )


        elif operation == "Data Imbalance": 
            st.subheader(operation)
            ini_col1,ini_col2 = st.columns([3.5,1.2])
            col1,col2 = ini_col1.columns([2,2])
            placeholder1 = col1.empty()
            placeholder2 = col2.empty()
            placeholder3 = col2.empty()
            selected_dataset = ini_col2.selectbox("Please select the Dataset", file_name, key='selected_dataset',   
                                                    index=file_name.index(st.session_state['default_name']))
            st.session_state['default_name'] = selected_dataset
            df = Manager.read_parquet(file_name=selected_dataset)
            selected_col = ini_col2.selectbox("Select the Column", Manager.get_imbalance_cols(df), key="imb_select_col")
            operation_to_use = ini_col2.radio(label="Select the Operation", 
                                                options=['SMOTE', 'Over/Under Sampling' ], 
                                                horizontal=False, label_visibility="collapsed")
            resample_disable = True
            if selected_col is not None:
            # Display pie chart, value counts, and the DataFrame
                placeholder1.plotly_chart(Manager.plot_imbalance_piechart(df, selected_col), 
                                            use_container_width=True)
                values_count = df[selected_col].value_counts()
                placeholder2.write(values_count)
                placeholder3.dataframe(df,hide_index = True )
                resample_disable = True
            else:
                ini_col1.info("The dataset is well-balanced, which is great for analysis!" , icon = "🔥")
                resample_disable = False 
                
           
            # Smote and over/under sampling techniques are used to resample the imbalance data
            if operation_to_use == "SMOTE":
                resample_disable = selected_col is None
                if ini_col2.button("Resample", key="resample_Smote" , type = "primary" , use_container_width = True , disabled= resample_disable):
                    # perform Smote resampling
                    resampled_df = Manager.smote_resampling(df, selected_col)
                    # Display resampled data, value counts, and options for updating, saving, and downloading
                    placeholder1.plotly_chart(Manager.plot_imbalance_piechart(resampled_df, selected_col), use_container_width=True)
                    values_count = resampled_df[selected_col].value_counts()
                    placeholder2.write(values_count)
                    placeholder3.dataframe(resampled_df,hide_index = True )
                    ini_col2.success("Operation performed successfully")
                    Manager.delete_pages_sessions()
                    # Options for updating, saving, and downloading
                    storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])    
                    with storage_tab1:
                        st.warning("Do you wish to overwrite the existing file")
                        if st.button("Update"):
                            try:
                                # update_parquet(df=resampled_df,file_name=selected_dataset)
                                Manager.update_parquet(df=resampled_df,file_name=selected_dataset)
                                Manager.modify_metadata(file_name=selected_dataset, new_process=[f"df = smote_resampling(df,'{selected_col}','{operation_to_use}')"])
                                Manager.delete_in_page_session()
                                time.sleep(5)
                                st.experimental_rerun()
                            except Exception as e:
                                st.exception(e)

                        with storage_tab2:
                            null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                            if len(null_update) > 0 or 'null_update' in st.session_state:
                                st.session_state.null_update = True

                                Manager.create_parquet(df=resampled_df,file_name=null_update)
                                Manager.delete_in_page_session()
                                time.sleep(3)
                                st.experimental_rerun()
                            else:
                                st.info('Please fill in inputs')

                        with storage_tab3:
                            csv = resampled_df.to_csv().encode('utf-8')
                            st.download_button(
                            label="Download",
                            data=csv,
                            file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                            mime='text/csv',
                            )

            else:     
            # Create a DataFrame for specifying class sampling options given by the user
                columns_data = []
                if selected_col:
                    for index, value in df[selected_col].value_counts().items():
                        column_info = {
                            "ClassName": index,
                            "Count": value,
                            "Null Values %": df[df[selected_col] == index][selected_col].isnull().sum().round(1) * 100,
                            "Select": False,
                            "Replacement": False,
                            "No_of_Samples": ""
                        }
                        columns_data.append(column_info)
                        info_df = pd.DataFrame(columns_data)

    
                    advance_df = ini_col2.data_editor(
                        info_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Select": st.column_config.CheckboxColumn("Select", default=False),
                            "No_of_Samples": st.column_config.NumberColumn("No of Samples"),
                            "Replacement": st.column_config.CheckboxColumn("Replacement", default=False),
                            })
                    if sum(advance_df["Select"]) > 1:
                        ini_col2.error("Select only one class at a time.")
                        resample_button_disabled = True  
                    else:   
                
                     resample_button_disabled = False  
                else:
                    pass
                resample_button_disabled = selected_col is None
                if ini_col2.button("Resample", key="resample_sampling", type="primary", use_container_width=True , disabled = resample_button_disabled) or "resample_btn" in st.session_state:
                    st.session_state.resample_btn = True
                # Display resampled data, value counts, and options for updating, saving, and downloading13
                    sampling_df , metadata = Manager.over_under_sampling(df ,  advance_df , selected_col)
                    placeholder1.plotly_chart(Manager.plot_imbalance_piechart(sampling_df, selected_col), use_container_width=True)
                    values_count = sampling_df[selected_col].value_counts()
                    placeholder2.write(values_count)
                    placeholder3.dataframe(sampling_df , use_container_width=True )
                    ini_col2.success("Resampling  done successfully")
                    storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                    with storage_tab1:
                        st.warning("Do you wish to overwrite the existing file")
                        if st.button("Update"):
                            try:
                                # update_parquet(df=sampling_df,file_name=selected_dataset)
                                Manager.update_parquet(df=remove_outliers,file_name=selected_dataset)
                                Manager.modify_metadata(file_name=selected_dataset, new_process=[metadata])
                                Manager.delete_in_page_session()
                                time.sleep(3)
                                st.experimental_rerun()
                            except Exception as e:
                                st.exception(e)
                    with storage_tab2:
                        null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                        if len(null_update) > 0 or 'null_update' in st.session_state:
                            st.session_state.null_update = True

                            Manager.create_parquet(df=remove_outliers,file_name=null_update)
                            Manager.delete_in_page_session()
                            time.sleep(3)
                            st.experimental_rerun()
                        else:
                            st.info('Please fill in inputs')

                    with storage_tab3:
                        csv = sampling_df.to_csv().encode('utf-8')
                        st.download_button(
                        label="Download",
                        data=csv,
                        file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                        mime='text/csv',
                        )
            # else:
            #     st.write("Please select the operation") 
        
        elif operation == "Data Transformation":
            st.subheader("Data Transformation")
        
            ini_col1, ini_col2 = st.columns([3.5,1.2])
            col1, col2  = ini_col1.columns([1,1.2])
            placeholder1 = col1.empty()
            placeholder2 = col2.empty()

            # Select the dataset
            selected_dataset = ini_col2.selectbox(
            "Select the Dataset",
            file_name,
            disabled = False,
            index=file_name.index(st.session_state.get('default_name', file_name[0])),
            )
            st.session_state.default_name = selected_dataset
        
            df = Manager.read_parquet(file_name=selected_dataset)
            if 'df' not in st.session_state:
               st.session_state.df = df
               
            placeholder1.dataframe(df[:100], use_container_width=True,
                                    hide_index=True  , height = 450)
            data_transformation_type = placeholder2.radio(
                label="Select the type of Transformation",
                options=["Aggregation", "Filtering", "Feature Engineering"],
                horizontal=True
            )

            if data_transformation_type == "Aggregation":
                 # Get a list of columns excluding datetime and timedelta columns
                selected_columns = df.select_dtypes( exclude=['datetime', 'timedelta' ] ).columns.tolist()
                groupby_by_col = ini_col2.selectbox("Select the Group By Column",selected_columns, key="groupby_col")
                # Allow the user to choose between aggregating a single column or the entire DataFrame
                groupby_to_perform = ini_col2.radio(label = "Groupby to apply:" , options = ["Single Column" , "Entire DataFrame"] , horizontal = True , label_visibility = "collapsed")
                if data_transformation_type == "Aggregation" and groupby_to_perform == "Single Column":
                    # Check if 'agg_df' session state exists, and if not, create it as an empty DataFrame
                    if 'agg_df' not in st.session_state:
                        st.session_state.agg_df = pd.DataFrame(columns=['Agg Column', 'Aggregation function'])
                    
                    # Display the aggregation configuration DataFrame
                    col2.dataframe(st.session_state.agg_df, use_container_width=True, hide_index=True ,height = 200)
                    
                    # Allow the user to select a column to perform aggregation on
                    groupby_by_agg_field = ini_col2.selectbox("Select Columns to Aggregate", list(set(df.columns) - set([groupby_by_col])), key="groupby_agg_fields")
                    
                    # Determine the available aggregation functions based on the selected column's data type
                    if groupby_by_agg_field in df.select_dtypes(include=['object', 'category']).columns:
                        option_list = c.CATEGORICAL_OPTIONS
                    else:
                        option_list = c.NUMERIC_OPTIONS
                    
                    # Allow the user to select multiple aggregation functions
                    groupby_by_agg_functions = ini_col2.multiselect("Select Aggregate function", option_list, key="groupby_agg_functions")
                    
                    # Check if any aggregations are selected, display info message if none selected
                    if len(st.session_state.agg_df) == 0:
                        ini_col2.info("Add the aggregations to be performed")
                        st.session_state.reset_disable = True
                    else:
                        # ini_col1.info("The following aggregations will be executed , Click the apply button!!")
                        st.session_state.reset_disable = False


                    # # Add the selected column and aggregation functions to 'agg_df'
                    if ini_col2.button("Add", type="primary", use_container_width=True):
                        value = {"Agg Column": [groupby_by_agg_field], "Aggregation function": [groupby_by_agg_functions]}
                        st.session_state.agg_df = pd.concat([st.session_state.agg_df, pd.DataFrame(value)], ignore_index=True)
                        ini_col1.success("Column added")
                        st.experimental_rerun()

                    # Apply the aggregation functions using 'agg_df'
                    if col2.button("Apply", type="secondary", use_container_width=False) or "agg_btn" in st.session_state:
                        st.session_state.agg_btn = True
                        if len(st.session_state.agg_df) > 0:
                            aggregated_df , metadata = Manager.perform_aggregation_Column(df, st.session_state.agg_df, groupby_by_col)
                            placeholder1.dataframe(aggregated_df, use_container_width=True, hide_index=True , height = 400)
                            ini_col2.success("Aggregation performed successfully")
                        # save  , update , download option 
                        storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                        with storage_tab1:
                            st.warning("Do you wish to overwrite the existing file")
                            if st.button("Update"):
                                try:
                                    # update_parquet(df=aggregated_df,file_name=selected_dataset)
                                    Manager.update_parquet(df=aggregated_df,file_name=selected_dataset)
                                    Manager.modify_metadata(file_name=selected_dataset, new_process=metadata)
                                    time.sleep(3)
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.exception(e)

                        with storage_tab2:
                            null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                            if len(null_update) > 0 or 'null_update' in st.session_state:
                                st.session_state.null_update = True

                                Manager.create_parquet(df=aggregated_df,file_name=null_update)
                                Manager.delete_in_page_session()
                                time.sleep(3)
                                st.experimental_rerun()
                            else:
                                st.info('Please fill in inputs')

                        with storage_tab3:
                                csv = aggregated_df.to_csv().encode('utf-8')
                                st.download_button(
                                label="Download",
                                data=csv,
                                file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                                mime='text/csv',
                                )

                    if col2.button("Reset", disabled=st.session_state.get('reset_disable', False)):
                        st.session_state.agg_df = 0
                        st.session_state.agg_df = pd.DataFrame(columns=['Agg Column', 'Aggregation function'])
                        st.session_state.reset_disable = True
                        st.experimental_rerun()
                    else:
                        st.session_state.reset_disable = False  
                    

                elif data_transformation_type == "Aggregation" and groupby_to_perform == "Entire DataFrame":
                    # Check if 'reset_disable' session state exists, and if not, initialize it as False
                    if 'reset_disable' not in st.session_state:
                        st.session_state.reset_disable = False

                    # Allow the user to select an aggregation function for the entire DataFrame
                    groupby_by_agg_functions = ini_col2.selectbox("Select Aggregate function", c.CATEGORICAL_OPTIONS + c.NUMERIC_OPTIONS)

                    # Display info message if the default aggregation function ('0') is selected
                    if groupby_by_agg_functions == '0':
                        ini_col2.info("You selected 0 (default value).")


                    if col2.button("Reset",disabled=st.session_state.reset_disable):
                            df=  placeholder1.dataframe(st.session_state.df , use_container_width = True , hide_index = True)
                            st.write(df)
                            st.session_state.reset_disable = True
                            st.experimental_rerun()
                    else:
                            st.session_state.reset_disable = False

                    # Apply aggregation to the entire DataFrame when the "Apply" button is clicked
                    if ini_col2.button(label="Apply", use_container_width=True, type="primary") or "agg_btn"  in st.session_state:
                        st.session_state.agg_btn = True
                        # Check the data type of the groupby column and perform aggregation accordingly
                        if groupby_by_col in df.select_dtypes(include=['object', 'category']).columns:
                            # Combine the groupby column and numeric columns for aggregation
                            numeric_columns = df.select_dtypes(include=np.number)
                            dfs_to_concat = [df[[groupby_by_col]], numeric_columns]
                            combined_df = pd.concat(dfs_to_concat, axis=1)
                            agg_dataframe = Manager.groupby_and_aggregate_DataFrame(combined_df, groupby_by_agg_functions, groupby_by_col)
                            placeholder1.dataframe(agg_dataframe, use_container_width=True)
                        elif groupby_by_col in df.select_dtypes(np.number).columns:
                            # Combine the groupby column and other numeric columns for aggregation
                            numeric_columns = df.select_dtypes(include=np.number).drop(columns=[groupby_by_col])
                            dfs_to_concat = [df[[groupby_by_col]], numeric_columns]
                            combined_df = pd.concat(dfs_to_concat, axis=1)
                            agg_dataframe = Manager.groupby_and_aggregate_DataFrame(combined_df, groupby_by_agg_functions, groupby_by_col)
                            placeholder1.dataframe(agg_dataframe, use_container_width=True)
                            ini_col2.success("Aggregation performed successfully")
                        else:
                            ini_col1.info("Please click the apply button to perform aggregation")


                            # save  , update , download option 
                        storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                        with storage_tab1:
                            st.warning("Do you wish to overwrite the existing file")
                            if st.button("Update"):
                                try:
                                    Manager.update_parquet(df=agg_dataframe,file_name=selected_dataset)
                                    Manager.modify_metadata(file_name=selected_dataset, new_process=[f"df = groupby_and_aggregate_DataFrame(df, '{groupby_by_agg_functions}', '{groupby_by_col}')"])
                                    Manager.delete_in_page_session()
                                    time.sleep(3)
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.exception(e)

                        with storage_tab2:
                            null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                            if len(null_update) > 0 or 'null_update' in st.session_state:
                                st.session_state.null_update = True

                                Manager.create_parquet(df=agg_dataframe,file_name=null_update)
                                time.sleep(3)
                                st.experimental_rerun()
                            else:
                                st.info('Please fill in inputs')

                            with storage_tab3:
                                csv = agg_dataframe.to_csv().encode('utf-8')
                                st.download_button(
                                label="Download",
                                data=csv,
                                file_name=f'Operated_{selected_dataset}_{datetime.datetime.now()}.csv',
                                mime='text/csv',
                                )
                
            elif data_transformation_type == "Filtering":
                selected_columns = df.select_dtypes( exclude=['datetime', 'timedelta' ] ).columns.tolist()
                # Select the groupby column and initialize filter DataFrame
                groupby_by_col = ini_col2.selectbox("Select the Group By Column",selected_columns, key="groupby_col")
                if 'filter_df' not in st.session_state:
                    st.session_state.filter_df = pd.DataFrame(columns=['Mapping','Condition'])
                filter_df = col2.dataframe(st.session_state.filter_df, use_container_width=True , hide_index = True , height = 200)
                # Determine available operators based on the data type of the groupby column
                if groupby_by_col in df.select_dtypes(include=['object', 'category']).columns:
                        operators = c.CATEGORICAL_OPTIONS
                else:
                        operators = c.NUMERIC_OPTIONS
                filter_condition = ini_col2.selectbox("Select Aggregate function", operators, key="groupby_agg_functions")

                filter_value = ini_col2.selectbox(label = "Value", options= df[groupby_by_col].unique() , key = "filtering_value")
                if isinstance(filter_value, (int, float)):
                    formatted_filter_value = filter_value
                else:
                    formatted_filter_value = f"'{filter_value}'"
                

                # Add filter condition when the "Add" button is clicked
                if ini_col2.button("Add", type="primary", use_container_width=True) or "add_filter " in st.session_state:
                    st.session_state.add_filter = True
                    new_row = {'Condition': f'{groupby_by_col} {c.NUMERIC_OPTIONS[filter_condition]} {str(formatted_filter_value)}'}
                    st.session_state.filter_df = pd.concat([st.session_state.filter_df, pd.DataFrame([new_row])], ignore_index=True)
                    st.session_state.filter_df['Mapping'] = st.session_state.filter_df.index.map(lambda x: f'c{x + 1}')
                    # print(st.session_state.filter_df)
                    st.experimental_rerun()


                condition_dict = {f'c{i+1}': condition for i, condition in enumerate(st.session_state.filter_df['Condition'])}
                condition_box = col2.text_input("Write the filter Condition (e.g., c1, c2, c3):" )
                # Validate input conditions and apply filtering when "Apply" button is clicked
                if col2.button("Apply", type="secondary" ) or "apply_filter" in st.session_state:
                    st.session_state.apply_filter = True
                    if len(condition_box) == 0:
                        col2.error("Please write the filter condition")
                    result_df  , metadata = Manager.apply_filter_condition(df , condition_box, condition_dict)
                    if result_df is not None:
                        ini_col2.success("Filter performed successfully")
                    else:
                        ini_col1.error("Invalid user input. Unable to evaluate the condition.Check the parenthesis")
                    placeholder1.dataframe(result_df, use_container_width = True , hide_index = True , height= 400 )
                    storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                    with storage_tab1:
                        st.warning("Do you wish to overwrite the existing file")
                        if st.button("Update"):
                            try:
                                Manager.update_parquet(df=result_df,file_name=selected_dataset)
                                Manager.modify_metadata(file_name=selected_dataset, new_process=metadata)
                                Manager.delete_in_page_session()
                                time.sleep(3)
                                st.experimental_rerun()
                            except Exception as e:
                                st.exception(e)
                    with storage_tab2:
                            null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                            if len(null_update) > 0 or 'null_update' in st.session_state:
                                st.session_state.null_update = True

                                Manager.create_parquet(df=result_df,file_name=null_update)
                                Manager.delete_in_page_session()
                                time.sleep(3)
                                st.experimental_rerun()
                            else:
                                st.info('Please fill in inputs')

                    with storage_tab3:
                            csv_data = result_df.to_csv(index=False, encoding='utf-8')
                        # Format the current datetime to include it in the file name
                            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            file_name = f'Operated_{filter_condition}_{datetime.now()}.csv'
                            st.write("Click here to Download")

                            # Use st.button to create a download button
                            st.download_button(
                                label="Download",
                                data=csv_data,
                                file_name=file_name,
                                mime='text/csv',
                            )   
                if col2.button("Reset", disabled=st.session_state.get('reset_disable', False)):
                    st.session_state.filter_df = 0
                    st.session_state.filter_df = pd.DataFrame(columns=['Mapping', 'Condition'])
                    st.session_state.reset_disable = True
                    st.experimental_rerun()
                    # Set condition_box to 0 here
                    condition_box = 0
                else:
                    st.session_state.reset_disable = False  
                if  st.session_state.filter_df.empty and len(condition_box) > 0:
                    col2.error("Please add the condition in the dataframe")
                    
            else:
                operations = col2.selectbox("Select the operation",options = ["Encoding", "Scaling", 'Shift/Roll', 'Arithmetic','Diff'])
                if operations == "Encoding":
                    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    selected_columns = categorical_columns
                    groupby_by_col = ini_col2.selectbox("Select the  Column", selected_columns, 
                                                        key="groupby_col1"  , help = "This operation is performed only on categorical columns")
                    encoding_type = ini_col2.selectbox("Select the type of encoding", 
                                                       options=['Label encoding', 'Ordinal encoding', 'One-hot Encoding'])
                    if encoding_type == "Ordinal encoding":
                        class_list = ini_col2.multiselect("Select the Class  :red[Please Add the class in increasing order of their importance (High weightage should be added first)]", df[groupby_by_col].unique())
                    else:
                        class_list = []
                    # Display a warning if using one-hot encoding on columns with many unique values
                    if encoding_type == "One-hot Encoding" and len(df[groupby_by_col].unique()) > 15:   
                        ini_col1.warning(f" Alert!! >> \nThe selected columns has **:red[{len(df[selected_columns].unique())} unique values]**. Are you sure you want to Proceed?")
                    # Handle the reset button to revert to the original DataFrame
                    try:   
                        if col2.button("Reset",disabled=st.session_state.reset_disable):
                            df=  placeholder1.dataframe(st.session_state.df , use_container_width = True , hide_index = True)
                            st.write(df)
                            st.session_state.reset_disable = True
                            st.experimental_rerun()

                        else:
                           st.session_state.reset_disable = False
                    except Exception as e:
                      print("The error in the reset button" ,  e)
                     # Apply encoding when the "Apply" button is clicked
                    if ini_col2.button("Apply" , use_container_width = True , type = "primary") or "encoding_btn" in st.session_state:
                        st.session_state.encoding_btn = True
                        encoded_df, encode_map = Manager.apply_encoding(df,groupby_by_col, encoding_type, class_list)
                        placeholder1.dataframe(encoded_df , use_container_width = True , hide_index = True)
                        # col2.write(encode_map)    
                        col2.success("Encoding operation is done successfully")
                        if "encoded_df" not in st.session_state:
                             st.session_state.encoded_df = df
                        
                        storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                        with storage_tab1:
                            st.warning("Do you wish to overwrite the existing file")
                            if st.button("Update"):
                                try:
                                    # update_parquet(df=encoded_df,file_name=selected_dataset)
                                    Manager.update_parquet(df=encoded_df,file_name=selected_dataset)
                                    Manager.modify_metadata(file_name=selected_dataset, new_process=[f"""df = apply_encoding(df, "{groupby_by_col}", "{encoding_type}", {class_list})"""])
                                    Manager.delete_in_page_session()
                                    time.sleep(3)
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.exception(e)
                        with storage_tab2:
                            null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                            if len(null_update) > 0 or 'null_update' in st.session_state:
                                st.session_state.null_update = True

                                Manager.create_parquet(df=encoded_df,file_name=null_update)
                                Manager.delete_in_page_session()
                                time.sleep(3)
                                st.experimental_rerun()
                            else:
                                st.info('Please fill in inputs')

                        with storage_tab3:
                            csv = encoded_df.to_csv().encode('utf-8')
                            st.download_button(
                            label="Download",
                            data=csv,
                            file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                            mime='text/csv',
                            )
               
                elif operations == "Scaling":
                    st.session_state.enable_columns = True
                    numerical_columns = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
                    groupby_by_col = ini_col2.selectbox("Select the Column", numerical_columns, key="groupby_col1" , help = "Scaling is performed only on numerical columns")
                    operation_perform = ini_col2.radio("Select the operation" , ["Single Column" , "Entire DataFrame"] ,key = "operation_type")
                    scaling_type = ini_col2.selectbox("Select the type of Scaling", options=['Standard Scaling', 'MinMax Scaling', 'Robust Scaling'])
                   
                    try:   
                        if col2.button("Reset",disabled=st.session_state.reset_disable):
                            df=  placeholder1.dataframe(st.session_state.df , use_container_width = True , hide_index = True )
                            st.write(df)
                            st.session_state.reset_disable = True
                            st.experimental_rerun()
                        else:
                           st.session_state.reset_disable = False
                    except Exception as e:
                      print("The error in the reset button" ,  e)
                  
                      # Apply scaling when the "Apply" button is clicked by calling the apply_scaling function 
                    if ini_col2.button("Apply" , type = "primary" , use_container_width= True) or "column_scale_btn" in st.session_state:
                        st.session_state.column_scale_btn = True
                        scaled_df = Manager.apply_scaling(df, scaling_type ,groupby_by_col, operation_perform)
                        placeholder1.dataframe(scaled_df , use_container_width = True , hide_index = True , height = 380)
                        col2.success("Scaling  performed successfully!")
                        print(f"apply_scaling(df , '{scaling_type}' , '{groupby_by_col}' , '{operation_perform}')")
                        storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                        with storage_tab1:
                            st.warning("Do you wish to overwrite the existing file")
                            if st.button("Update"):
                                try:
                                    # update_parquet(df=scaled_df,file_name=selected_dataset)
                                    Manager.update_parquet(df=scaled_df,file_name=selected_dataset)
                                    Manager.modify_metadata(file_name=selected_dataset, new_process=[f"df = apply_scaling(df , '{scaling_type}' , '{groupby_by_col}', '{operation_perform}')"])
                                    Manager.delete_in_page_session()
                                    time.sleep(3)
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.exception(e)

                        with storage_tab2:
                            null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                            if len(null_update) > 0 or 'null_update' in st.session_state:
                                st.session_state.null_update = True
                                Manager.create_parquet(df=scaled_df,file_name=null_update)
                                Manager.delete_in_page_session()
                                time.sleep(3)
                                st.experimental_rerun()
                            else:
                                st.info('Please fill in inputs')

                        with storage_tab3:
                            csv = scaled_df.to_csv().encode('utf-8')
                            st.download_button(
                            label="Download",
                            data=csv,
                            file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                            mime='text/csv',
                            )

                elif operations == "Shift/Roll":
                    # Get a list of numerical columns for shift/roll operations
                    numerical_columns = df.select_dtypes(exclude=['object', 'category']).columns.tolist()

                    # Select the column for shift/roll operations
                    selected_columns = ini_col2.selectbox("Select the Column", numerical_columns, key="groupby_col1", help="This operation is performed only on numerical values")

                    # Select the type of operation (Rolling or Shifting)
                    op_type = ini_col2.selectbox("Selected type of Operation:", ['Rolling (Moving Avg)', 'Shifting (Leading/Lagging)'])
                    try:
                        # Handle the reset button to revert to the original DataFrame
                        if col2.button("Reset", disabled=st.session_state.reset_disable):
                            df = placeholder1.dataframe(st.session_state.df, use_container_width=True, hide_index=True ,height = 500)
                            st.write(df)
                            st.session_state.reset_disable = True
                            st.experimental_rerun()
                        else:
                            st.session_state.reset_disable = False
                    except Exception as e:
                        print("Error in the reset button:", e)

                    # Configure options based on the selected operation type
                    if op_type == "Rolling (Moving Avg)":
                        window_size = ini_col2.number_input("Window size", step=1, value=0)
                        min_periods = ini_col2.number_input("Enter min Periods", step=1, value=0)
                        conditions = ini_col2.selectbox(label="Condition", options=c.NUMERIC_OPTIONS)
                        is_center = ini_col2.checkbox("Center?")
                         # Apply rolling or shifting operation based on the selected type
                        if ini_col2.button("Apply", type="primary", use_container_width=True , key = "apply_rolling") or "rolling_btn" in st.session_state:
                            st.session_state.rolling_btn = True
                            if op_type == "Rolling (Moving Avg)" and window_size > 0:
                                rolling_df = Manager.apply_rolling(df, selected_columns, window_size, min_periods, conditions, is_center)
                                placeholder1.dataframe(rolling_df, use_container_width=True, hide_index=True , height = 350 )
                                col2.success("Rolling operation is performed successfully")
                            storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                            with storage_tab1:
                                st.warning("Do you wish to overwrite the existing file")
                                if st.button("Update"):
                                    try:
                                        Manager.update_parquet(df=rolling_df,file_name=selected_dataset)
                                        Manager.modify_metadata(file_name=selected_dataset, new_process=[f"df=apply_rolling(df , '{selected_columns}', {window_size}, {min_periods}, '{conditions}', {is_center})"])
                                        Manager.delete_in_page_session()
                                        time.sleep(3)
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.exception(e)

                            with storage_tab2:
                                null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                                if len(null_update) > 0 or 'null_update' in st.session_state:
                                    st.session_state.null_update = True
                                    Manager.create_parquet(df=rolling_df,file_name=null_update)
                                    Manager.delete_in_page_session()
                                    time.sleep(3)
                                    st.experimental_rerun()
                                else:
                                    st.info('Please fill in inputs')

                            with storage_tab3:
                                    csv = rolling_df.to_csv().encode('utf-8')
                                    st.download_button(
                                    label="Download",
                                    data=csv,
                                    file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                                    mime='text/csv',
                                    )
                   
                    elif op_type == "Shifting (Leading/Lagging)":
                        shift_period = ini_col2.number_input("Select Shifting Period", step=1)
                        if ini_col2.button("Apply", type="primary", use_container_width=True , key = "apply_shifting") or "shift_btn" in st.session_state:
                            st.session_state.shift_btn = True
                            if op_type == "Shifting (Leading/Lagging)" and shift_period > 0:
                                shift_df = Manager.apply_shifting(df, selected_columns, shift_period )
                                placeholder1.dataframe(shift_df, use_container_width=True, hide_index=True)
                                col2.success("Shifting operation is performed successfully")
                            else:
                                ini_col1.info("Select the operation")
                            storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                            with storage_tab1:
                                st.warning("Do you wish to overwrite the existing file")
                                if st.button("Update"):
                                    try:
                                        Manager.update_parquet(df=shift_df,file_name=selected_dataset)
                                        Manager.modify_metadata(file_name=selected_dataset, new_process=[f"df = apply_shifting(df, '{selected_columns}', {shift_period})"])
                                        Manager.delete_in_page_session()
                                        time.sleep(3)
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.exception(e)

                            with storage_tab2:
                                null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                                if len(null_update) > 0 or 'null_update' in st.session_state:
                                    st.session_state.null_update = True
                                    Manager.create_parquet(df=shift_df,file_name=null_update)
                                    Manager.delete_in_page_session()
                                    time.sleep(3)
                                    st.experimental_rerun()
                                else:
                                    st.info('Please fill in inputs')
                            with storage_tab3:
                                csv = shift_df.to_csv().encode('utf-8')
                                st.download_button(
                                label="Download",
                                data=csv,
                                file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                                mime='text/csv',
                                )
                    else:
                        ini_col1.info("Select the operation ")

                elif operations =="Arithmetic":

                    if 'check' not in st.session_state:
                        st.session_state['check'] = df._get_numeric_data().columns[0]
                        st.session_state['arithmetic'] = "trigonometric"
                     
                    arithmetic_col = ini_col2.selectbox("Select the Column", df._get_numeric_data().columns, key='arithmetic_col')
                    
                    arithmetic_condition = ini_col2.selectbox("Select the Operation", ['trigonometric', 'power/root', 'logarithm'], key='arithmetic_condition')
                    if st.session_state['check'] != arithmetic_col:
                        st.session_state['check'] = arithmetic_col
                        st.session_state["math_operation_btn"] = False
                    elif st.session_state['arithmetic'] != arithmetic_condition:
                        st.session_state['arithmetic'] = arithmetic_condition
                        st.session_state["math_operation_btn"] = False

                    if arithmetic_condition == 'trigonometric':
                        selected_functions = ini_col2.multiselect("Select the function", c.TRIGNOMETRIC_OPTIONS)
                    elif arithmetic_condition == 'power/root':
                        power_to = ini_col2.number_input("Select Power" , step = 1)
                    else:
                        log_function = ini_col2.selectbox("Select the log function", ["loge", "log10"])

                    try:
                        if col2.button("Reset",disabled=st.session_state.reset_disable):
                            df=  placeholder1.dataframe(st.session_state.df , use_container_width = True , hide_index = True)
                            st.write(df)
                            st.session_state.reset_disable = True
                            st.experimental_rerun()

                        else:
                            st.session_state.reset_disable = False
                    except Exception as e:
                         print("The error in the reset button" ,  e)
                        
                    if ini_col2.button("Apply", type="primary", use_container_width=True) or st.session_state["math_operation_btn"] == True:
                        st.session_state.math_operation_btn= True  
                    
                        if arithmetic_condition == 'trigonometric':
                            if len(selected_functions) > 0:
                                arithmetic_df = Manager.apply_arithmetic_operation(df, arithmetic_col, arithmetic_condition, selected_functions=selected_functions)
                                st.session_state["meta_data"] = f"df = apply_arithmetic_operation(df, '{arithmetic_col}', '{arithmetic_condition}', selected_functions={selected_functions})"
                                height = 450  
                                col2.success("Trigonometric Operation is performed successfully")
                            else:
                                col2.error("Please select the function")

                        elif arithmetic_condition == 'power/root':
                            if power_to > 0:
                                arithmetic_df = Manager.apply_arithmetic_operation(df, arithmetic_col, arithmetic_condition, power_to=power_to)
                                st.session_state["meta_data"] = f"df = apply_arithmetic_operation(df, '{arithmetic_col}', '{arithmetic_condition}', power_to={power_to})"
                                height = 450 
                                col2.success("Operation is performed successfully")
                            else:
                                col2.error("Please give the power")

                        elif arithmetic_condition == 'logarithm':   
                            arithmetic_df = Manager.apply_arithmetic_operation(df, arithmetic_col, arithmetic_condition, log_function = log_function )
                            st.session_state["meta_data"] = f"""df = apply_arithmetic_operation(df, '{arithmetic_col}', '{arithmetic_condition}', log_function = '{log_function}' )"""
                            height = 450 
                            col2.success("log operation is performed successfully")

                        else:
                            col2.info("Select the operation to be performed")

                        if 'arithmetic_df' in locals():
                            placeholder1.dataframe(arithmetic_df, use_container_width=True, hide_index=True, height=height)
                        # print(f"""apply_arithmetic_operation(df, '{arithmetic_col}', '{arithmetic_condition}', selected_functions='{selected_functions}', power_to={power_to}, log_function='{log_function}')")
                       
                        storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                        with storage_tab1:
                            st.warning("Do you wish to overwrite the existing file")
                            if st.button("Update"):
                            
                                try:
                                    # update_parquet(df=arithmetic_df,file_name=selected_dataset)
                                    Manager.update_parquet(df=arithmetic_df,file_name=selected_dataset)
                                    Manager.modify_metadata(file_name=selected_dataset, new_process=[st.session_state["meta_data"]])
                                    Manager.delete_in_page_session()
                                    time.sleep(3)
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.exception(e)

                        with storage_tab2:
                            null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                            if len(null_update) > 0 or 'null_update' in st.session_state:
                                st.session_state.null_update = True

                                Manager.create_parquet(df=arithmetic_df,file_name=null_update)
                                Manager.delete_in_page_session()
                                time.sleep(3)
                                st.experimental_rerun()
                            else:
                                st.info('Please fill in inputs')

                        with storage_tab3:
                                csv = arithmetic_df.to_csv().encode('utf-8')
                                st.download_button(
                                label="Download",
                                data=csv,
                                file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                                mime='text/csv',
                                )

                else:   
                    # Select the numerical column for the difference operation
                    diff_col = ini_col2.selectbox("Select the Column for Operation", df._get_numeric_data().columns, key='diff_roll_col')
                    # Select the range for calculating the difference
                    select_range = ini_col2.number_input("Please select the range", min_value=1, max_value=len(df), step=1)
                    try:
                        # Handle the reset button to revert to the original DataFrame
                        if col2.button("Reset", disabled=st.session_state.reset_disable):
                            df = placeholder1.dataframe(st.session_state.df, use_container_width=True, hide_index=True)
                            st.write(df)
                            st.session_state.reset_disable = True
                            st.experimental_rerun()
                        else:
                            st.session_state.reset_disable = False
                    except Exception as e:
                        print("Error in the reset button:", e)
                    # Apply the difference operation to the selected column with the specified range
                    if ini_col2.button("Apply", type="primary", use_container_width=True) or "diff_btn" in st.session_state:
                        st.session_state.diff_btn = True
                        diff_df  = Manager.apply_diff_function(df, diff_col, select_range)
                        placeholder1.dataframe(diff_df, use_container_width=True, hide_index=True)
                        col2.success("Diff Operation is performed successfully")
                        storage_tab1,storage_tab2,storage_tab3 = ini_col2.tabs(['Update','Save','Download'])
                        with storage_tab1:
                            st.warning("Do you wish to overwrite the existing file")
                            if st.button("Update"):
                                try:
                                    Manager.update_parquet(df=diff_df,file_name=selected_dataset)
                                    Manager.modify_metadata(file_name=selected_dataset, new_process=[f"df = apply_diff_function(df, {diff_col}, {select_range})"])
                                    Manager.delete_in_page_session()
                                    time.sleep(3)
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.exception(e)

                        with storage_tab2:
                            null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension")

                            if len(null_update) > 0 or 'null_update' in st.session_state:
                                st.session_state.null_update = True

                                Manager.create_parquet(df=diff_df,file_name=null_update)
                                Manager.delete_in_page_session()
                                time.sleep(3)
                            else:
                                st.info('Please fill in inputs')

                        with storage_tab3:
                            csv = diff_df.to_csv().encode('utf-8')
                            st.download_button(
                            label="Download",
                            data=csv,
                            file_name=f'Operated_{selected_dataset}_{datetime.now()}.csv',
                            mime='text/csv',
                            )           
        else:
         st.error("Please select the operation")
                 
except Exception as  e:
      print("The error in the code is ",e)