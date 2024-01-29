import time 

import pandas as pd
import streamlit as st
import constants as c
import iosense_connect as io

import Utilities.py_tools as Manager
from datetime import datetime

st.set_page_config(layout="wide",page_title="Data Import",page_icon="https://storage.googleapis.com/ai-workbench/Data%20Import.svg")

def main():
    st.subheader('Data Import')
    import_select = st.radio("select",["Upload", "Connect"],horizontal=True,label_visibility='collapsed')
    st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)
    
    Manager.faclon_logo()

    if import_select == 'Upload':

        st.subheader('Upload file')
        uploaded_files = st.file_uploader("Upload file", 
                                          type=["csv", "xlsx", "xls"],
                                          accept_multiple_files=True,
                                          label_visibility="collapsed")

        if uploaded_files:
            for file in uploaded_files:
                file_extension = file.name.split(".")[-1].lower()
                try:
                    df = pd.read_csv(file) if file_extension == "csv" else pd.read_excel(file)
                    if df.empty:
                        st.warning('Empty Dataframe. Operation Denied')
                        time.sleep(3)
                        break
                    st.dataframe(df[:50],hide_index=True,use_container_width=True)
                except Exception as e:
                    st.error(f"Operation Failed....{e}")
                    break

                for column in df.columns:
                    try:
                        if df[column].dtype == 'object':
                            df[column] = pd.to_datetime(df[column])
                        else:
                            df[column] = pd.to_numeric(df[column])
                    except Exception as e:
                        continue   

                Manager.create_parquet(df=df,file_name=file.name)


    else:
        st.subheader('Connect to IO-Sense')
        col1,col2 = st.columns([3.5,1.5])

        # user_key = col1.text_input("User ID", label_visibility= 'collapsed')
        # iosense_button = col2.button("Execute")

        # if user_key:
        #     if iosense_button or 'iosense_userid_execute' in st.session_state:
        #         st.session_state['iosense_userid_execute'] = True

        user_key = c.API_KEY
        try:
            values = Manager.verify_userid_iosense(user_key=user_key)
            if not values.empty:
                col1.success('Api Authentication Successful')
                value = values['devID']
                io_sense = io.DataAccess(user_key,c.URL,c.CONTAINER)
                with col2.container():
                    radio_select = st.radio("Select: ",['Single-Choice','Multi-Choice'],horizontal=True)

                    if radio_select == 'Single-Choice':
                        with col2.container():
                            selected_option = st.selectbox('Select Device', value, key="da_device_select")
                            sensor_list = []
                            sensors = io_sense.get_device_metadata(device_id=selected_option)
                            for sensor in sensors["sensors"]:
                                sensor_list.append(sensor["sensorId"])
                            select_sensors = st.multiselect("Select sensor",sensor_list,key="da_sensors_select")
                            col_1,col_2 = st.columns([2,2])
                            start_time = col_1.date_input("Select Start Date",key="da_start_time")
                            end_time = col_2.date_input("Select End Date",key="da_end_time")
                            if start_time <= end_time:
                                period = end_time - start_time
                            else:
                                period = start_time - end_time
                            period = period.days
                
                            sub_col1,sub_col2,sub_col3 = st.columns(3)
                            cal = sub_col1.checkbox("cal")
                            db = sub_col2.checkbox("gcs", value=True)
                            ist = sub_col3.checkbox("IST",value=True)

                        if len(select_sensors) == 0:
                            select_sensors=None

                        if db:
                            db='gcs'
                        else:
                            db=None

                        df = Manager.iosense_multi_select_concatinator(io_sense,
                                                            [selected_option],
                                                            select_sensors,
                                                            start_time,
                                                            end_time,
                                                            db,
                                                            cal,
                                                            ist)

                        col1.dataframe(df.head(50),hide_index=True,use_container_width=True,height=435)
                    else:
                        df = pd.DataFrame()

                        with col2.container():
                            
                            dev_prefixes = set()
                            for item in values['devTypeID']:
                                # prefix = item.split('_')[0]
                                dev_prefixes.add(item)

                            selected_option = st.selectbox('Select Device Type', dev_prefixes, key="da_prefix_select")
                            filtered_devices = values[values['devTypeID'] == selected_option]['devID'].to_list()
                            selected_option = st.multiselect('Present Devices', filtered_devices, key="da_cluster_devices_select")

                            if len(selected_option)>0:
                                sensor_list = []
                                sensors = io_sense.get_device_metadata(device_id=selected_option[0])
                                for sensor in sensors["sensors"]:
                                    sensor_list.append(sensor["sensorId"])
                                select_sensors = st.multiselect("Select sensor",sensor_list,key="da_cluster_sensors_select")
                                col_1,col_2 = st.columns([2,2])
                                start_time = col_1.date_input("Select Start Date",key="da_cluster_start_time")
                                end_time = col_2.date_input("Select End Date",key="da_cluster_end_time")
                                if start_time <= end_time:
                                   period = end_time - start_time
                                else:
                                    period = start_time - end_time
                                period = period.days
                                sub_col1,sub_col2,sub_col3 = st.columns(3)
                                cal = sub_col1.checkbox("cal")
                                db = sub_col2.checkbox("gcs", value=True)
                                ist = sub_col3.checkbox("IST",value=True)
                                if len(select_sensors) == 0:
                                    select_sensors=None
                                st.write(db)

                                if db:
                                    db="gcs"
                                    st.write(db)

                                else:
                                    db=None


                                df = Manager.iosense_multi_select_concatinator(io_sense,
                                                                    selected_option,
                                                                    select_sensors,
                                                                    start_time,
                                                                    end_time,
                                                                    db,
                                                                    cal,
                                                                    ist)

                                col1.dataframe(df[:50],hide_index=True,use_container_width=True,height=435)

                            else:
                                st.info('Please select devices')

                if len(df)>0 and df is not None:
                    sub_tabs1,sub_tabs2 = col2.tabs(['Save','Download'])

                    with sub_tabs1:
                        iosense_save = st.text_input("Please Provide a Name",help="Please press enter to save \n Dont add .csv or any other extension")
                        if len(iosense_save) > 0 or 'iosense_save' in st.session_state:
                            st.session_state.iosense_save = True
                            Manager.create_parquet(df=df,file_name=iosense_save)
                            Manager.invoke_iosense(iosense_save , selected_option, select_sensors , start_time , end_time , period , cal , db , ist)
                            Manager.delete_in_page_session()
                        else:
                            st.info('Please fill in inputs')
                    with sub_tabs2:
                        csv = df.to_csv().encode('utf-8')
                        st.download_button(
                        label="Download",
                        data=csv,
                        file_name=f'{selected_option}_{datetime.now()}.csv',
                        mime='text/csv',
                        use_container_width=True,
                        type="primary"
                        )
                else:
                    st.toast('Authentication Failed')
                    col1.error("No Devices Added on IO Sense.")
        
        except Exception as e:
            col1.exception(e)              

if __name__ == "__main__":
    main()