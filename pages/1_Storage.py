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
st.set_page_config(layout="wide",page_title="Storage",page_icon="https://storage.googleapis.com/ai-workbench/Storage.svg")
Manager.faclon_logo()

st.subheader('Storage')
file_name = Manager.files_details()
select_lvl1 = st.radio("Yooo",["Data Insights","Null Operations","Merge","Concat","Resample","Files Details"],horizontal=True,label_visibility="collapsed")
st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)

if len(file_name) > 1:

    if 'default_name' not in st.session_state or st.session_state.default_name not in file_name:
        st.session_state.default_name = file_name[0]

    if select_lvl1 == "Data Insights":
        st.subheader("Insights", help="This section provides insights on the displayed data, including the column name, data type, percentage of null values, category status (yes or no), count, and an option to perform a drop operation via the checkbox.")
        col1,col2 = st.columns([3.5,1.5])

        tab0_name = col2.selectbox("Please select file:",file_name,key="tab0_name",index=file_name.index(st.session_state['default_name']))
        st.session_state['default_name'] = tab0_name

        df = Manager.read_parquet(file_name=tab0_name)
        tab0_categories = Manager.determine_categorial_columns(df=df,threshold=0.03)

        tab0_info = Manager.get_table(df,tab0_categories)

        tab0_info_df = pd.DataFrame(tab0_info)
        tab0_info_df['checkbox'] = False
        test = col1.empty()
        tab0_final_df = test.data_editor(
            tab0_info_df.round(1),
            use_container_width=True,
            height=320,
            hide_index=True,
            column_config={
                "checkbox": st.column_config.CheckboxColumn(
                "checkbox",
                default=False,
                help="On selecting checkbox, you can drop these columns"
            )
            })

        if True in tab0_final_df['checkbox'].values:
            if col2.button('Drop',use_container_width=True,type="primary") or 'stage1_drop' in st.session_state:
                st.session_state.stage1_drop = True
                cols_to_drop = tab0_final_df[tab0_final_df['checkbox'].values]['Column Name'].to_list()
                df = df.drop(cols_to_drop,axis=1)
                col1.code(f"""{cols_to_drop}""")

                storage_tab1,storage_tab2,storage_tab3 = col2.tabs(['Update','Save','Download'])

                with storage_tab1:
                    st.warning("Do you wish to overwrite the existing file")
                    if st.button("Update",key="random_update"):
                        try:
                            Manager.update_parquet(df=df,file_name=tab0_name)
                            Manager.modify_metadata(file_name=tab0_name,new_process=[f"df = df.drop(columns={cols_to_drop},axis=1)"])
                            Manager.delete_pages_sessions(key='default_name')
                            time.sleep(5)
                        except Exception as e:
                            st.exception(e)

                with storage_tab2:
                    null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension",key="random_key")

                    if len(null_update) > 0:
                        Manager.create_parquet(file_name=null_update)
                        Manager.delete_in_page_session()
                        time.sleep(5)
                        st.experimental_rerun()
                    else:
                        st.info('Please fill in inputs')

                with storage_tab3:
                    csv = df.to_csv().encode('utf-8')
                    st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f'Operated_{tab0_name}_{datetime.now()}.csv',
                    mime='text/csv',
                    use_container_width=True,
                    type="primary"
                    )

    elif select_lvl1 == "Null Operations":
        st.subheader("Nullifier",help="You can fill null values using this functionality")
        st.session_state.flag = False

        if 'agg_df' not in st.session_state:
            st.session_state.agg_df = pd.DataFrame(columns=['Column','Aggregation','Key'])

        col1,col2 = st.columns([3.5,1.5])

        # col1.info("Please Select Columns to perform Fill Null Operation")
        selected_csv = col2.selectbox("Please Select CSV",file_name,disabled=False,index=file_name.index(st.session_state['default_name']))
        
        if st.session_state['default_name'] != selected_csv:
            st.session_state.default_name = selected_csv
            st.session_state.agg_df = pd.DataFrame(columns=['Column','Aggregation','Key'])
            st.experimental_rerun()

        df = Manager.read_parquet(file_name=selected_csv)
        if len(df.columns[df.isnull().any()].tolist()) != 0:
            operation = col2.radio("Select: ",['Individual','DataFrame'],horizontal=True)

            if operation == "Individual":
                selected_columns = col2.selectbox("Please select column",df.columns[df.isnull().any()].tolist())

                if np.issubdtype(df[selected_columns].dtype, np.number):
                    numeric_agg = col2.selectbox("Agg Numeric Method:",['0','ffill','bfill','Mean','Median','Min','Max','Custom fill'])
                    key="numeric"
                    value = {"Column": selected_columns,"Aggregation":numeric_agg,"Key":"numeric"}
                    st.session_state.append_disable = False

                elif df[selected_columns].dtype == 'object':
                    object_agg = col2.selectbox("Agg Object Method",['ffill','bfill','Max Frequency','Min Frequency'])
                    key="object"
                    value = {"Column": selected_columns,"Aggregation": object_agg,"Key":"object"}
                    st.session_state.append_disable = False

                else:
                    st.session_state.append_disable = True
                
                if len(st.session_state.agg_df) == 0:
                    col1.info("Please select aggregations to be performed")
                    st.session_state.reset_disable = True
                else:
                    col1.success("The following will be executed")
                    st.session_state.reset_disable = False
                
                col1_1,col1_2 = col1.columns([3.5,1.5])

                col1_2.dataframe(st.session_state.agg_df,use_container_width=True,hide_index=True)

                if col1_2.button("Reset",disabled=st.session_state.reset_disable):
                    st.session_state.agg_df = 0
                    st.session_state.agg_df = pd.DataFrame(columns=['Column','Aggregation',"Key"])
                    st.experimental_rerun()


                if col2.button("Append",disabled=st.session_state.append_disable):
                        col1_2.success("Column Appended")    
                        st.session_state.agg_df = pd.concat([st.session_state.agg_df, pd.DataFrame([value])],ignore_index=True)
                        st.experimental_rerun()

                if len(st.session_state.agg_df) == 0:
                    col1_1.dataframe(df,use_container_width=True,hide_index=True)
                else:
                    final_df = pd.DataFrame()
                    info = st.session_state['agg_df'].to_dict()
                    df,st.session_state.metadata_store = Manager.perform_individual_null_operation(df,st.session_state.agg_df)
                    col1_1.dataframe(df.round(1)[:100],use_container_width=True,hide_index=True)

            else:
                col1.warning("The Displayed Dataframe is Upto first 100 rows")
                st.session_state.agg_df = pd.DataFrame(columns=['Column','Aggregation','Key'])
                agg_select = col2.selectbox("Select Method: ",['Default','0','Ffill','Bfill',
                                                               'Custom String', 'Custom Integer'])
                st.session_state.flag = False

                if agg_select == 'Ffill':
                    st.session_state.flag = True
                    df.fillna(method='ffill',inplace=True)
                    st.session_state.metadata_store = [f"df=df.fillna(method='ffill',inplace=True)"]
                elif agg_select == 'Bfill':  
                    st.session_state.flag = True
                    df.fillna(method='bfill', inplace=True)
                    st.session_state.metadata_store = [f"df=df.fillna(method='bfill',inplace=True)"]
                elif agg_select=='0':
                    st.session_state.flag = True
                    df.fillna(0,inplace=True)
                    st.session_state.metadata_store = [f"df=df.fillna(method=0,inplace=True)"]
                elif agg_select == 'Mean':  
                    st.session_state.flag = True
                    df.fillna(df.mean(), inplace=True)
                    st.session_state.metadata_store = [f"df=df.fillna(df.mean()),inplace=True)"]
                elif agg_select == 'Median':
                    st.session_state.flag = True
                    df.fillna(df.median(), inplace=True)
                    st.session_state.metadata_store = [f"df=df.fillna(df.median()),inplace=True)"]
                elif agg_select == 'Min':
                    st.session_state.flag = True
                    df.fillna(df.min(), inplace=True)
                    st.session_state.metadata_store = [f"df=df.fillna(df.min()),inplace=True)"]
                elif agg_select == 'Max':
                    st.session_state.flag = True
                    df.fillna(df.max(), inplace=True)
                    st.session_state.metadata_store = [f"df=df.fillna(df.max()),inplace=True)"]
                elif agg_select == 'Custom String':
                    fill_value = col2.text_input("Please type in string")
                    if len(fill_value) > 0:
                        st.session_state.flag = True
                        df.fillna(str(fill_value),inplace=True)
                        st.session_state.metadata_store = [f"df=df.fillna({str(fill_value)}),inplace=True)"]
                elif agg_select == 'Custom Integer':
                    fill_value = col2.text_input("Please type in numeric value")
                    if len(fill_value) > 0:
                        try:
                            fill_value = float(fill_value)
                            df.fillna(fill_value,inplace=True)
                            st.session_state.flag = True
                            st.session_state.metadata_store = [f"df=df.fillna({float(fill_value)}),inplace=True)"]
                        except Exception as e:
                            col2.error('Invalid Integer Type')
                            st.session_state.flag = False
                else:
                    col2.info("Please select any operation")

                col1.dataframe(df[:200],use_container_width=True,hide_index=True)

        else:
            col1.text("          ")
            col1.error("No Null Values found")
        
        if len(st.session_state.agg_df) > 0 or st.session_state.flag is True:
            tab1_1,tab1_2,tab1_3 = col2.tabs(['Update','Save','Download'])
            with tab1_1:
                st.warning("Do you wish to overwrite the existing file")
                if st.button("Update",key="null_updates"):
                    try:
                        Manager.update_parquet(df=df,file_name=selected_csv)
                        Manager.modify_metadata(file_name=selected_csv,new_process=st.session_state.metadata_store)
                        Manager.delete_pages_sessions(key='default_name')
                        time.sleep(5)
                    except Exception as e:
                        st.exception(e)

            with tab1_2:
                null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension",key="null_key")

                if len(null_update) > 0:
                    Manager.create_parquet(file_name=null_update)
                    Manager.delete_in_page_session()
                    time.sleep(5)
                    st.experimental_rerun()
                else:
                    st.info('Please fill in inputs')

            with tab1_3:
                csv = df.to_csv().encode('utf-8')
                st.download_button(
                label="Download",
                data=csv,
                file_name=f'Operated_{selected_csv}_{datetime.now()}.csv',
                mime='text/csv',
                use_container_width=True,
                type="primary"
                )

    elif select_lvl1=="Merge":
        
        if 'merge_flag' not in st.session_state:
            st.session_state.tab2_flag = False

        st.subheader("MergeXform",help="You can perform merge operations on selected")

        col1,col2 = st.columns([3,2])
        merge_tab1,merge_tab2 = col2.columns(2)

        dataframe1_name = merge_tab1.selectbox("Select dataframe:",file_name,index=file_name.index(st.session_state['default_name']))
        st.session_state['default_name'] = dataframe1_name
        remaining_files = [file for file in file_name if file != dataframe1_name]
        dataframe2_name = merge_tab2.selectbox("Select dataframe:",remaining_files,key="tab5_name_selection")

        dataframe1 = Manager.read_parquet(file_name=dataframe1_name)
        dataframe2 = Manager.read_parquet(file_name=dataframe2_name)

        dataframe1_columns = merge_tab1.selectbox("Select Column Name:",dataframe1.columns,key="tab5_left_column")
        dataframe2_columns = merge_tab2.selectbox("Select Column Name:",dataframe2.columns,key="tab5_right_column")

        storage_merge_select = col2.radio("Please select method",
                                          ('Inner','Outer','Left','Right'),
                                          horizontal=True)

        if storage_merge_select == "Inner":
            how="inner"
        elif storage_merge_select == "Outer":
            how="outer"
        elif storage_merge_select == "Left":
            how="left"
        else:
            how="right"

        try:
            merge_df = Manager.get_merged_df(dataframe1,dataframe2,
                                dataframe1_columns,dataframe2_columns,
                                how)
                        
            col1.success("The selected columns can be merged")
            col1.dataframe(merge_df[:200],hide_index=True,use_container_width=True,height=390)
            st.session_state.merge_flag = True

        except Exception as e:
            col1.error(f"Reason: {e}")
            st.session_state.merge_flag = False

        if  st.session_state.merge_flag is True:
            tab2_sub1,tab2_sub2 = col2.tabs(['Save','Download'])

            with tab2_sub1:
                null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension",key="merge_key")

                if len(null_update) > 0:
                    Manager.create_parquet(df=merge_df,file_name=null_update)
                    Manager.delete_in_page_session()
                    time.sleep(5)
                    st.experimental_rerun()
                else:
                    st.info('Please fill in inputs')

            with tab2_sub2:
                csv = merge_df.to_csv().encode('utf-8')
                st.download_button(
                label="Download",
                data=csv,
                file_name=f'Operated_{dataframe1_name}_{datetime.now()}.csv',
                mime='text/csv',
                use_container_width=True,
                type="primary"
                )

    elif select_lvl1=="Concat":
        st.subheader("Concatinator",help="Helps you concat multiple Dataframe with axis=0")
        col1,col2 = st.columns([3,2])
        storage_concat = col2.multiselect("Select Files to Concat:",file_name)

        if len(storage_concat) == 0:
            col1.text("            ")
            col1.info("Please select files to carry on the operation")
        
        elif len(storage_concat) == 1:
            col1.warning("Cant Concat One Cs")
            col1.dataframe(Manager.read_parquet(file_name=storage_concat[0])[:200],use_container_width=True,hide_index=True)

        else:
            
            concat_df = Manager.concatinator(storage_concat)

            col1.success("The selected files can be concated")
            col1.dataframe(concat_df[:200],hide_index=True,use_container_width=True)

            storage_tabs1,storage_tabs2 = col2.tabs(['Save','Download'])

            with storage_tabs1:
                null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension",key="concat_key")

                if len(null_update) > 0:
                    Manager.create_parquet(df=concat_df,file_name=null_update)
                    Manager.delete_in_page_session()
                    time.sleep(5)
                    st.experimental_rerun()
                else:
                    st.info('Please fill in inputs')

            with storage_tabs2:
                csv = concat_df.to_csv().encode('utf-8')
                st.download_button(
                label="Download",
                data=csv,
                file_name=f'Operated_{storage_concat}_{datetime.now()}.csv',
                mime='text/csv',
                use_container_width=True,
                type="primary"
                )

    elif select_lvl1=="Resample":

        st.subheader("Resampler",help="This page helps you resample your time to the desired timedelta")

        col1,col2 = st.columns([3,2])
        col1.text("        ")
        resample_select = col2.selectbox("Select File:",file_name,index=file_name.index(st.session_state['default_name']))
        st.session_state['default_name'] = resample_select

        df = Manager.read_parquet(file_name=resample_select)

        if any(df.dtypes == 'datetime64[ns]'):
            col1.success("The following file can be resampled")

            datetime_columns = df.columns[df.dtypes == 'datetime64[ns]']
            resample_column = col2.selectbox("Resample On:",datetime_columns)

            try:
                aggregation_mode = col2.selectbox("Aggregation mode:",['first','last','mean','min','max'])
                df.set_index(df[resample_column],inplace=True)

                resample_option = col2.radio('Select resample option', ['Second', 'Minute', 'Hour', 'Day','Month','Year'],horizontal=True)

                if resample_option == 'Second':
                    key='S'
                    frequency = col2.select_slider('Select Second',list(range(1,60)),value=30,label_visibility="collapsed")
                    # frequency = col2.number_input("Select seconds",min_value=1,max_value=59,step=1,value=30         )

                elif resample_option == 'Minute':
                    key='T'
                    frequency = col2.select_slider('Select Minute',list(range(1,60)),label_visibility="collapsed")

                elif resample_option == 'Hour':
                    key='H'
                    frequency = col2.select_slider('Select Hour',list(range(1,24)),label_visibility="collapsed")

                elif resample_option == 'Day':
                    key='D'
                    frequency = col2.select_slider('Select Day',list(range(1,29)),label_visibility="collapsed")
                elif resample_option == 'Month':
                    key='M'
                    frequency = col2.select_slider('Select Day',list(range(1,29)),label_visibility="collapsed")
                elif resample_option == 'Year':
                    key='Y'
                    frequency = col2.select_slider('Select Day',list(range(1,5)),label_visibility="collapsed")

                resampled_df = Manager.resampler(df,frequency,key,aggregation_mode)
                col1.dataframe(resampled_df[:200],use_container_width=True,height=500)            
                
                tab4_sub1,tab4_sub2,tab4_sub3 = col2.tabs(['Update','Save','Download'])


                with tab4_sub1:
                    st.warning("Do you wish to overwrite the existing file")
                    if st.button("Update",key="resample_update"):
                        try:
                            Manager.update_parquet(df=df,file_name=resample_select)
                            Manager.modify_metadata(file_name=resample_select,new_process=[f"df=df.set_index(df['{resample_column}'],inplace=True)",
                                                             f"df=resampler(df,'{frequency}','{key}','{aggregation_mode}')"])
                            Manager.delete_pages_sessions(key='default_name')
                            time.sleep(5)
                        except Exception as e:
                            st.exception(e)

                with tab4_sub2:
                    null_update = st.text_input("Please Provide Name",help="Please press enter to save \n Dont add .csv or any other extension",key="resample_key")

                    if len(null_update) > 0:
                        Manager.create_parquet(file_name=null_update)
                        Manager.delete_in_page_session()
                        time.sleep(5)
                        st.experimental_rerun()
                    else:
                        st.info('Please fill in inputs')

                with tab4_sub3:
                    csv = df.to_csv().encode('utf-8')
                    st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f'Operated_{resample_select}_{datetime.now()}.csv',
                    mime='text/csv',
                    use_container_width=True,
                    type="primary"
                    )

            except Exception as e:
                col1.error(f'{e}')    
    
        else:
            col1.warning('Cant perform operation since datetime doesnt exist')

    else:

        st.subheader("Obliterate")
        col1,col2 = st.columns([3.5,1.5])
        col1.info("This section consist of all the files details with Name,Size,Created and Modified. If you wish to delete unwanted files then please select any of the checkbox to continue the operation.")

        def get_all_details_files():
            csv_files = glob.glob(c.DEFAULT_STORAGE.format(file="*"))

            file_size = []
            file_modified = []
            file_time = []

            try:
                for file in csv_files:
                    file_size.append(os.path.getsize(file) / (1024*1024))
                    file_time.append(os.path.getctime(file))
                    file_modified.append(os.path.getmtime(file))
            except Exception as e:
                pass
            return file_size,file_modified,file_time

        file_size,file_modified,file_time = get_all_details_files()

        delete_df = pd.DataFrame({
            'File Name': file_name,
            'File Size (MB)': file_size,
            'File Created Time': file_modified,
            'File Modified Time': file_time,
            'Delete Box': False
        })

        delete_df['File Created Time'] = delete_df['File Created Time'].apply(lambda x: c.BASE_DATE + timedelta(seconds=x))
        delete_df['File Modified Time'] = delete_df['File Modified Time'].apply(lambda x: c.BASE_DATE + timedelta(seconds=x))

        detailed_df = col1.data_editor(
            delete_df.round(2)[:100],
            use_container_width=True,
            height=320,
            column_config={
                "Delete Box": st.column_config.CheckboxColumn(
                "Delete Box",
                default=False
            )
            })
        
        if True in detailed_df['Delete Box'].values:
            col2.warning("""This Action will delete all the selected files.""")                
            files_to_delete = detailed_df[detailed_df['Delete Box'] == True]['File Name'].to_list()
            if  col2.button("Delete",type="primary",use_container_width=True) or 'tab5_delete_state' in st.session_state:
                st.session_state["tab5_delete_state"] = True

                col2.warning("Please Note: This Action is irreversible")

                if  col2.button("Execute",type="primary",use_container_width=True) or 'delete_authenticate' in st.session_state:
                    st.session_state['delete_authenticate'] = True
                    col2.success("Done Successfully!!")

                    for name in files_to_delete:
                        delete_path = c.DEFAULT_STORAGE.format(file=name)
                        delete_meta_data_path = c.DEFAULT_METADATA.format(file=name)
                        os.remove(delete_path)
                        os.remove(delete_meta_data_path)

                    Manager.delete_in_page_session()
                    time.sleep(5)
                    st.experimental_rerun()
   
else:
    st.error("Please Import CSVs")