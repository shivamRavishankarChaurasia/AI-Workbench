import time
import pandas as pd
import streamlit as st
import constants as c
import Utilities.py_tools as Manager
from datetime import datetime

st.set_page_config(
    layout="wide",
    page_title="Data Import",
    page_icon="https://img.icons8.com/?size=100&id=86319&format=png&color=000000"
)
Manager.faclon_logo()

def main():
    """
    Main function to handle various data import methods through a Streamlit interface.
    Provides options for importing data from files, APIs, databases, web scraping, cloud storage,
    public datasets, IoT devices, and more. Handles user input and processes data accordingly.
    """
    st.subheader('Data Import')
    import_select = st.radio(
        "Select", ["Upload", "API", "Database", "Web-Scraping", "Cloud", "Public Datasets", "IoT device", "FTP", "Google Sheets"],
        horizontal=True,
        label_visibility='collapsed'
    )
    st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)
    
    # Upload Method
    if import_select == 'Upload':
        handle_upload()

    # API Method
    elif import_select == "API":
        handle_api()

    # Database Method
    elif import_select == "Database":
        handle_database()

    # Web Scraping Method
    elif import_select == "Web-Scraping":
        handle_web_scraping()

    # Cloud Method
    elif import_select == "Cloud":
        handle_cloud()

    # Public Datasets Method
    elif import_select == "Public Datasets":
        handle_public_datasets()

    # IoT Device Method
    elif import_select == "IoT device":
        handle_iot_device()

    # FTP Method
    elif import_select == "FTP":
        handle_ftp()

    # Google Sheets Method
    elif import_select == "Google Sheets":
        handle_google_sheets()

def handle_upload():
    st.subheader('Upload File')
    uploaded_files = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for file in uploaded_files:
            file_extension = file.name.split(".")[-1].lower()
            try:
                df = pd.read_csv(file) if file_extension == "csv" else pd.read_excel(file)
                if df.empty:
                    st.warning('Empty DataFrame. Operation Denied')
                    time.sleep(3)
                    break
                st.dataframe(df.head(50), hide_index=True, use_container_width=True)

                # Clean and process data
                df, metadata = Manager.validate_and_clean_data(df)
                Manager.create_parquet(df=df, file_name=file.name)

            except Exception as e:
                st.error(f"File processing failed: {e}")

def handle_api():
    st.subheader("API Data Fetch")
    end_point = st.text_input("Enter the API Endpoint")
    api_key = st.text_input("Enter the API Key", type="password")
    fetch_button = st.button("Fetch Data")

    if fetch_button:
        try:
            if not end_point or not api_key:
                st.warning("Please provide both endpoint and API key.")
            else:
                data = Manager.fetch_data_from_api(end_point, api_key)
                if data:
                    df = Manager.process_data(data)
                    if df is not None and not df.empty:
                        st.dataframe(df.head(50), hide_index=True, use_container_width=True)
                        Manager.create_parquet(df=df, file_name="api_data.parquet")
                    else:
                        st.error("Fetched data is empty or invalid.")
                else:
                    st.error("Failed to fetch data from the API.")
        except Exception as e:
            st.error(f"API Fetch failed: {e}")

def handle_database():
    st.subheader("Database Query")
    db_type = st.selectbox("Select Database Type", ["SQL", "NoSQL"])
    db_connection = st.text_input("Enter Database Connection String")
    query_or_command = st.text_area("Enter SQL Query or NoSQL Command")
    execute_button = st.button("Execute")

    if execute_button:
        try:
            if not db_connection or not query_or_command:
                st.warning("Please provide both connection string and query/command.")
            else:
                if db_type == "SQL":
                    df = Manager.fetch_data_from_sql_database(db_connection, query_or_command)
                else:
                    df = Manager.fetch_data_from_nosql_database(db_connection, query_or_command)

                if df is not None and not df.empty:
                    st.dataframe(df.head(50), hide_index=True, use_container_width=True)
                    Manager.create_parquet(df=df, file_name=f"{db_type.lower()}_database_data.parquet")
                else:
                    st.error("Query/Command returned no results.")
        except Exception as e:
            st.error(f"Database operation failed: {e}")

def handle_web_scraping():
    st.subheader("Web Scraping")
    url = st.text_input("Enter the URL to scrape")
    scrape_button = st.button("Scrape Data")

    if scrape_button:
        try:
            if not url:
                st.warning("Please provide a URL.")
            else:
                data = Manager.scrape_data_from_web(url)
                if data:
                    df = Manager.process_data(data)
                    if df is not None and not df.empty:
                        st.dataframe(df.head(50), hide_index=True, use_container_width=True)
                        Manager.create_parquet(df=df, file_name="web_scraped_data.parquet")
                    else:
                        st.error("Scraped data is empty or invalid.")
                else:
                    st.error("Failed to scrape data or the website structure may have changed.")
        except Exception as e:
            st.error(f"Web scraping failed: {e}")

def handle_cloud():
    st.subheader("Cloud Data Access")
    cloud_provider = st.selectbox("Select Cloud Provider", ["AWS", "Azure", "Google Cloud"])
    bucket_name = st.text_input("Enter Bucket Name")
    file_key = st.text_input("Enter File Key")
    fetch_button = st.button("Fetch Cloud Data")

    if fetch_button:
        try:
            if not bucket_name or not file_key:
                st.warning("Please provide bucket name and file key.")
            else:
                df = Manager.fetch_data_from_cloud(cloud_provider, bucket_name, file_key)
                if df is not None and not df.empty:
                    st.dataframe(df.head(50), hide_index=True, use_container_width=True)
                    Manager.create_parquet(df=df, file_name="cloud_data.parquet")
                else:
                    st.error("No data found in the specified file.")
        except Exception as e:
            st.error(f"Cloud Data Fetch failed: {e}")

def handle_public_datasets():
    st.subheader("Public Dataset Access")
    dataset_url = st.text_input("Enter Public Dataset URL")
    load_button = st.button("Load Dataset")

    if load_button:
        try:
            if not dataset_url:
                st.warning("Please provide a dataset URL.")
            else:
                df = Manager.fetch_data_from_public_dataset(dataset_url)
                if df is not None and not df.empty:
                    st.dataframe(df.head(50), hide_index=True, use_container_width=True)
                    Manager.create_parquet(df=df, file_name="public_dataset.parquet")
                else:
                    st.error("Failed to fetch or process the dataset.")
        except Exception as e:
            st.error(f"Dataset loading failed: {e}")

def handle_iot_device():
    st.subheader("IoT Device Data Fetch")
    device_id = st.text_input("Enter Device ID")
    fetch_button = st.button("Fetch IoT Data")

    if fetch_button:
        try:
            if not device_id:
                st.warning("Please provide a device ID.")
            else:
                df = Manager.fetch_data_from_iot_device(device_id)
                if df is not None and not df.empty:
                    st.dataframe(df.head(50), hide_index=True, use_container_width=True)
                    Manager.create_parquet(df=df, file_name="iot_data.parquet")
                else:
                    st.error("No data found for the specified device.")
        except Exception as e:
            st.error(f"IoT Data Fetch failed: {e}")

def handle_ftp():
    st.subheader("FTP Data Fetch")
    ftp_url = st.text_input("Enter FTP URL")
    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password", type="password")
    fetch_button = st.button("Fetch FTP Data")

    if fetch_button:
        try:
            if not ftp_url or not username or not password:
                st.warning("Please provide FTP URL, username, and password.")
            else:
                df = Manager.fetch_data_from_ftp(ftp_url, username, password)
                if df is not None and not df.empty:
                    st.dataframe(df.head(50), hide_index=True, use_container_width=True)
                    Manager.create_parquet(df=df, file_name="ftp_data.parquet")
                else:
                    st.error("Failed to fetch data from FTP.")
        except Exception as e:
            st.error(f"FTP Data Fetch failed: {e}")

def handle_google_sheets():
    st.subheader("Google Sheets Data Fetch")
    sheet_url = st.text_input("Enter Google Sheets URL")
    fetch_button = st.button("Fetch Google Sheets Data")

    if fetch_button:
        try:
            if not sheet_url:
                st.warning("Please provide a Google Sheets URL.")
            else:
                df = Manager.fetch_data_from_google_sheets(sheet_url)
                if df is not None and not df.empty:
                    st.dataframe(df.head(50), hide_index=True, use_container_width=True)
                    Manager.create_parquet(df=df, file_name="google_sheets_data.parquet")
                else:
                    st.error("Failed to fetch data from Google Sheets.")
        except Exception as e:
            st.error(f"Google Sheets Data Fetch failed: {e}")

if __name__ == "__main__":
    main()
