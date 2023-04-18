import pandas as pd
import streamlit as st
#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
#import pandas_profiling
from PIL import Image
from streamlit_pandas_profiling import st_profile_report
#from pandas_profiling import ProfileReport
import numpy as np
#import pickle
from utilities import utils as utl
from views import metadata as mt
import os
import pyodbc
import sqlalchemy as sa


# @st.cache
def load_view():

    st.markdown('\n')
    st.markdown('\n')
    st.title('Data Upload')
    # Add the expander to provide some information about the app
    with st.sidebar.expander("About the page"):
        st.write("""
            User needs to upload the new claims dataset in excel/csv format which need to be predicted as true/flase claim.
              \n\nThe app has the minimum mode (recommended) and the complete code.
               The complete code includes more sophisticated analysis such as correlation analysis or interactions between
                variables which may requires expensive computations. )
         """)
        
    option = st.selectbox(
     'Choose specific domain : ',
     ('Health insurance', 'Auto insurance', 'Life insurance','General insurance'))
    if option == 'Health insurance': 

        if st.sidebar.button('Delete existing data'):
            # File name
            file = 'main_data.csv'
            # Path
            path = os.path.join('data', file)
            if os.path.exists(path):
                # Remove the file
                # 'file.txt'
                os.remove(path)

        sidebar_expander_data_upload = st.sidebar.expander("Select source for data upload",True)


        with sidebar_expander_data_upload:
 
            data_source = st.selectbox("Data Sources", ["Local","MS SQL", "MySQL","OracleSQL","AWS"])


    

    # Code to read a single file
        if data_source == "Local" :
            if 'main_data.csv' not in os.listdir('data'):
                st.markdown(""" <style> .font {
                font-size:35px ; font-family: 'Helvetica Neue'; color: #FCB216;} 
                </style> """, unsafe_allow_html=True)
                st.markdown('<p class="font">Upload your csv/excel data here</p>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("🗂️ Choose a file", type=['csv', 'xlsx'], key = 'load_data1')
                global data
                if uploaded_file is not None:
                    try:
                        data = pd.read_csv(uploaded_file)
                    except Exception as e:
                        print(e)
                        data = pd.read_excel(uploaded_file)

                    if st.button("Load Data", key = 'load_data'):
                        data.to_csv(r'data/main_data.csv', index=False)

            else :
                    data = pd.read_csv(r'data/main_data.csv')

        elif data_source == "MS SQL":
            sidebar_expander_mssql = st.sidebar.expander("DB Input",True)
            with sidebar_expander_mssql: 
                server_input = st.text_input("Enter MSSQL server name",key= 'key1')
                st.session_state.key1
                db_input = st.text_input("Enter Database name",key= 'key2')
                st.session_state.key2
                table_input = st.text_input("Enter table name",key= 'key3')
                st.session_state.key3
            username = 'admin'
            password = 'Digitalit123'

            if bool(server_input) and bool(db_input) and bool(table_input):
                cnxn_str_temp = ('Driver={SQL Server};SERVER='+server_input+';DATABASE='+db_input+';uid='+username+';pwd='+ password)
                if st.sidebar.button('Connect DB'):
                    print(cnxn_str_temp)
                    conn = pyodbc.connect(cnxn_str_temp)
                    query_input = ("SELECT * FROM "+table_input)
                    print(query_input)
                    try:
                        data = pd.read_sql(str(query_input), conn)
                        data.to_csv(r'data/main_data.csv', index=False)
                    except Exception as e:
                        print(e)
                        st.warning('Database details entered are not correct. Please recheck')
        
        elif data_source == "Oracle SQL" :
            sidebar_expander_oraclesql = st.sidebar.expander("DB Input",True)
            with sidebar_expander_oraclesql: 
                server_input = st.text_input("Enter Oracle SQL server name")
                db_input = st.text_input("Enter Database name")
                table_input = st.text_input("Enter table name")

    

    #if st.button("Load Data", key = 'load_data'):
        '''st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Helvetica Neue'; color: #FCB216;} 
             </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Upload your data here</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("🗂️ Choose a file", type=['csv', 'xlsx'], key = 'load_data2')
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                print(e)
                data = pd.read_excel(uploaded_file)'''

        if 'main_data.csv'  in os.listdir('data'): 
            data.to_csv(r'data/main_data.csv', index=False)
            st.markdown("## 1. Head of the data")
        # Raw data 
            st.dataframe(data)

        #Shape of the raw data
            st.markdown("## 2. Shape of the data")
            st.markdown('''Number of records''')
            st.write(data.shape[0])

            st.markdown("Number of Columns")
            st.write(data.shape[1])
            st.write("\n")
            st.write("\n")
            st.markdown("## 3. Data information")
            st.dataframe(data.describe())

        # Collect the categorical and numerical columns 
        
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
            columns_df = pd.read_csv('data/metadata/column_type_desc.csv')
            st.markdown("\n")
            st.markdown("## 4. Column types")
            st.dataframe(columns_df)

            option1 = st.sidebar.radio(
            'What variables do you want to include in the report?',
            ('All variables', 'A subset of variables'))

            if option1 == 'All Variables':
                data = data

            elif option1 == 'A subset of variables':
                var_list = list(data.columns)
                option3 = st.sidebar.multiselect(
                'Select variable(s) you want to include in the report.',
                    var_list)
                data = data[option3]

            option2 = st.sidebar.selectbox(
            'Choose Minimal Mode or Complete Mode',
            ('Minimal Mode', 'Complete Mode'))

            if option2 == 'Complete Mode':
                mode = 'complete'
                st.sidebar.warning(
                'The default minimal mode disables expensive computations such as '
                'correlations and duplicate row detection. Switching to complete mode '
                'may cause the app to run overtime or fail for large datasets due to computational limit.')
            elif option2 == 'Minimal Mode':
                mode = 'minimal'

        
            with st.expander("Change MetaData here"):
                st.markdown("### Change MetaData here")
            # Load the uploaded data
                if 'main_data.csv' not in os.listdir('data'):
                    st.markdown("Please upload data through `Upload Data` page!")
                else:
                    data = pd.read_csv('data/main_data.csv')

                # Read the column meta data for this dataset
                    col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')

                    st.markdown("#### Change the information about column types")

                # Use two column technique
                    col1, col2 = st.columns(2)

                    global name, type
                # Design column 1
                    name = col1.selectbox("Select Column", data.columns)

                # Design column two
                    current_type = col_metadata[col_metadata['column_name'] == name]['type'].values[0]

                    print(current_type)
                    column_options = ['numerical', 'categorical', 'object']
                    current_index = column_options.index(current_type)

                    type = col2.selectbox("Select Column Type", options=column_options, index=current_index)

                    st.write("""Select your column name and the new type from the data.
                                            To submit all the changes, click on *Submit changes* """)

                    if st.button("Change Column Type"):
                    # Set the value in the metadata and resave the file
                        col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')

                        col_metadata.loc[col_metadata['column_name'] == name, 'type'] = type
                        col_metadata.to_csv(r'data/metadata/column_type_desc.csv', index=False)
                        st.write("Your changes have been made!")
                        st.dataframe(col_metadata[col_metadata['column_name'] == name])

                    st.markdown(
                    'After we import the data, made the required changes in the meta data '
                    'we are checking the information about '
                    'the data frame, such as number of rows and columns,'
                    ' data types for each column, etc.')


            with st.expander("Profiling report generation"):
                if st.button('Generate Report'):
                    if mode == 'complete':
                        pr = data.profile_report(title="User uploaded table",
                                                progress_bar=True,
                                                dataset={
                                                "description": 'This profiling report was generated by CBSL',
                                                "copyright_holder": 'CBSL',
                                                "copyright_year": '2022'
                                            })
                        st_profile_report(pr)
                        export = pr.to_html()
                        st.download_button(label='📥 Download Profile Report', data=export, file_name='Profile Report')
                    elif mode == 'minimal':
                        pr = data.profile_report(minimal=True, title="User uploaded table",
                                                progress_bar=True,
                                                dataset={
                                                "description": 'This profiling report was generated by CBSL',
                                                "copyright_holder": 'CBSL',
                                                "copyright_year": '2022'
                                            })
                        st_profile_report(pr)
                        export = pr.to_html()
                        st.download_button(label='📥 Download Profile Report', data=export, file_name='Profile Report')
    elif option == 'Auto insurance':
        if st.sidebar.button('Delete existing data'):
            # File name
            file = 'fraud_oracle.csv'
            # Path
            path = os.path.join('data', file)
            if os.path.exists(path):
                # Remove the file
                # 'file.txt'
                os.remove(path)

        sidebar_expander_data_upload = st.sidebar.expander("Select source for data upload")


        with sidebar_expander_data_upload:
 
            data_source = st.selectbox("Data Sources", ["Local","MS SQL", "MySQL","OracleSQL","AWS"])


    

    # Code to read a single file
        if data_source == "Local" :
            if 'fraud_oracle.csv' not in os.listdir('data'):
                st.markdown(""" <style> .font {
                    font-size:35px ; font-family: 'Helvetica Neue'; color: #FCB216;} 
                    </style> """, unsafe_allow_html=True)
                st.markdown('<p class="font">Upload your csv/excel data here</p>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("🗂️ Choose a file", type=['csv', 'xlsx'], key = 'load_data1')
                
                if uploaded_file is not None:
                    try:
                        data = pd.read_csv(uploaded_file)
                    except Exception as e:
                        print(e)
                        data = pd.read_excel(uploaded_file)

                    if st.button("Load Data", key = 'load_data'):
                        data.to_csv(r'data/fraud_oracle.csv', index=False)

            else :
                data = pd.read_csv('data/fraud_oracle.csv')

        elif data_source == "MS SQL":
            sidebar_expander_mssql = st.sidebar.expander("DB Input",True)
            with sidebar_expander_mssql: 
                server_input = st.text_input("Enter MSSQL server name",key= 'key1')
                st.session_state.key1
                db_input = st.text_input("Enter Database name",key= 'key2')
                st.session_state.key2
                table_input = st.text_input("Enter table name",key= 'key3')
                st.session_state.key3
            username = 'admin'
            password = 'Digitalit123'

            if bool(server_input) and bool(db_input) and bool(table_input):
                cnxn_str_temp = ('Driver={SQL Server};SERVER='+server_input+';DATABASE='+db_input+';uid='+username+';pwd='+ password)
                if st.sidebar.button('Connect DB'):
                    print(cnxn_str_temp)
                    conn = pyodbc.connect(cnxn_str_temp)
                    query_input = ("SELECT * FROM "+table_input)
                    print(query_input)
                    try:
                        data = pd.read_sql(str(query_input), conn)
                        data.to_csv(r'data/fraud_oracle.csv', index=False)
                    except Exception as e:
                        print(e)
                        st.warning('Database details entered are not correct. Please recheck')
        
        elif data_source == "Oracle SQL" :
            sidebar_expander_oraclesql = st.sidebar.expander("DB Input",True)
            with sidebar_expander_oraclesql: 
                server_input = st.text_input("Enter Oracle SQL server name")
                db_input = st.text_input("Enter Database name")
                table_input = st.text_input("Enter table name")

    

    #if st.button("Load Data", key = 'load_data'):
        '''st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Helvetica Neue'; color: #FCB216;} 
             </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Upload your data here</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("🗂️ Choose a file", type=['csv', 'xlsx'], key = 'load_data2')
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                print(e)
                data = pd.read_excel(uploaded_file)'''

        if 'fraud_oracle.csv'  in os.listdir('data'): 
            data.to_csv(r'data/fraud_oracle.csv', index=False)
            st.markdown("## 1. Head of the data")
            # Raw data 
            st.dataframe(data)

            #Shape of the raw data
            st.markdown("## 2. Shape of the data")
            st.markdown('''Number of records''')
            st.write(data.shape[0])

            st.markdown("Number of Columns")
            st.write(data.shape[1])
            st.write("\n")
            st.write("\n")
            st.markdown("## 3. Data information")
            st.dataframe(data.describe())

        # Collect the categorical and numerical columns 
        
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
            columns_df = pd.read_csv(r'data/metadata/coltypecsv.csv')
            st.markdown("\n")
            st.markdown("## 4. Column types")
            st.dataframe(columns_df)

            option1 = st.sidebar.radio(
                'What variables do you want to include in the report?',
                ('All variables', 'A subset of variables'))

            if option1 == 'All Variables':
                data = data

            elif option1 == 'A subset of variables':
                var_list = list(data.columns)
                option3 = st.sidebar.multiselect(
                    'Select variable(s) you want to include in the report.',
                    var_list)
                data = data[option3]

            option2 = st.sidebar.selectbox(
                'Choose Minimal Mode or Complete Mode',
                ('Minimal Mode', 'Complete Mode'))

            if option2 == 'Complete Mode':
                mode = 'complete'
                st.sidebar.warning(
                    'The default minimal mode disables expensive computations such as '
                    'correlations and duplicate row detection. Switching to complete mode '
                    'may cause the app to run overtime or fail for large datasets due to computational limit.')
            elif option2 == 'Minimal Mode':
                mode = 'minimal'

        
            #with st.expander("Change MetaData here"):
                st.markdown("### Change MetaData here")
                # Load the uploaded data
                if 'fraud_oracle' not in os.listdir('data'):
                    st.markdown("Please upload data through `Upload Data` page!")
                else:
                    data = pd.read_csv('data/fraud_oracle.csv')

                    # Read the column meta data for this dataset
                    col_metadata = pd.read_csv(r'data/metadata/coltypecsv.csv')

                    st.markdown("#### Change the information about column types")

                    # Use two column technique
                    col1, col2 = st.columns(2)

                
                    # Design column 1
                    name = col1.selectbox("Select Column", data.columns)

                    # Design column two
                    current_type = col_metadata[col_metadata['column_name'] == name]['type'].values[0]

                # print(current_type)
                    column_options = ['numerical', 'categorical', 'object']
                    current_index = column_options.index(current_type)

                    type = col2.selectbox("Select Column Type", options=column_options, index=current_index)

                    st.write("""Select your column name and the new type from the data.
                                            To submit all the changes, click on *Submit changes* """)

                    if st.button("Change Column Type"):
                    # Set the value in the metadata and resave the file
                        col_metadata = pd.read_csv('data/metadata/coltypecsv.csv')

                        col_metadata.loc[col_metadata['column_name'] == name, 'type'] = type
                        col_metadata.to_csv('data/metadata/coltypecsv.csv', index=False)
                        st.write("Your changes have been made!")
                        st.dataframe(col_metadata[col_metadata['column_name'] == name])

                    st.markdown(
                        'After we import the data, made the required changes in the meta data '
                        'we are checking the information about '
                        'the data frame, such as number of rows and columns,'
                        ' data types for each column, etc.')


            with st.expander("Profiling report generation"):
                if st.button('Generate Report'):
                    if mode == 'complete':
                        pr = data.profile_report(title="User uploaded table",
                                                progress_bar=True,
                                                dataset={
                                                    "description": 'This profiling report was generated by CBSL',
                                                    "copyright_holder": 'CBSL',
                                                    "copyright_year": '2022'
                                                })
                        st_profile_report(pr)
                        export = pr.to_html()
                        st.download_button(label='📥 Download Profile Report', data=export, file_name='Profile Report')
                    elif mode == 'minimal':
                        pr = data.profile_report(minimal=True, title="User uploaded table",
                                                progress_bar=True,
                                                dataset={
                                                    "description": 'This profiling report was generated by CBSL',
                                                    "copyright_holder": 'CBSL',
                                                    "copyright_year": '2022'
                                                })
                        st_profile_report(pr)
                        export = pr.to_html()
                        st.download_button(label='📥 Download Profile Report', data=export, file_name='Profile Report')
'''else :
            data = pd.read_csv('data/main_data.csv')
            st.markdown("## 1. Head of the data")
            # Raw data 
            st.dataframe(data)

            #Shape of the raw data
            st.markdown("## 2. Shape of the data")
            st.markdown(''''''Number of records'''''')
            st.write(data.shape[0])

            st.markdown("Number of Columns")
            st.write(data.shape[1])
            st.write("\n")
            st.write("\n")
            st.markdown("## 3. Data information")
            st.dataframe(data.describe())

            # Collect the categorical and numerical columns 
            
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
            columns_df = pd.read_csv('data/metadata/column_type_desc.csv')
            st.markdown("\n")
            st.markdown("## 4. Column types")
            st.dataframe(columns_df)

            option1 = st.sidebar.radio(
                'What variables do you want to include in the report?',
                ('All variables', 'A subset of variables'))

            if option1 == 'All Variables':
                data = data

            elif option1 == 'A subset of variables':
                var_list = list(data.columns)
                option3 = st.sidebar.multiselect(
                    'Select variable(s) you want to include in the report.',
                    var_list)
                data = data[option3]

            option2 = st.sidebar.selectbox(
                'Choose Minimal Mode or Complete Mode',
                ('Minimal Mode', 'Complete Mode'))

            if option2 == 'Complete Mode':
                mode = 'complete'
                st.sidebar.warning(
                    'The default minimal mode disables expensive computations such as '
                    'correlations and duplicate row detection. Switching to complete mode '
                    'may cause the app to run overtime or fail for large datasets due to computational limit.')
            elif option2 == 'Minimal Mode':
                mode = 'minimal'

            
            with st.expander("Change MetaData here"):
                    st.markdown("### Change MetaData here")

                    # Read the column meta data for this dataset
                    col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')

                    st.markdown("#### Change the information about column types")

                    # Use two column technique
                    col1, col2 = st.columns(2)

                    global name1, type1
                    # Design column 1
                    name1 = col1.selectbox("Select Column", data.columns)

                    # Design column two
                    current_type = col_metadata[col_metadata['column_name'] == name1]['type'].values[0]

                    print(current_type)
                    column_options = ['numerical', 'categorical', 'object']
                    current_index = column_options.index(current_type)

                    type1 = col2.selectbox("Select Column Type", options=column_options, index=current_index)

                    st.write("""Select your column name and the new type from the data.
                                                To submit all the changes, click on *Submit changes* """)

                    if st.button("Change Column Type"):
                        # Set the value in the metadata and resave the file
                        col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')

                        col_metadata.loc[col_metadata['column_name'] == name, 'type'] = type
                        col_metadata.to_csv('data/metadata/column_type_desc.csv', index=False)
                        st.write("Your changes have been made!")
                        st.dataframe(col_metadata[col_metadata['column_name'] == name])

                    st.markdown(
                        'After we import the data, made the required changes in the meta data '
                        'we are checking the information about '
                        'the data frame, such as number of rows and columns,'
                        ' data types for each column, etc.')


            with st.expander("Profiling report generation"):
                if st.button('Generate Report'):
                    if mode == 'complete':
                        pr = data.profile_report(title="User uploaded table",
                                                progress_bar=True,
                                                dataset={
                                                    "description": 'This profiling report was generated by CBSL',
                                                    "copyright_holder": 'CBSL',
                                                    "copyright_year": '2022'
                                                })
                        st_profile_report(pr)
                        export = pr.to_html()
                        st.download_button(label='📥 Download Profile Report', data=export, file_name='Profile Report')
                    elif mode == 'minimal':
                        pr = data.profile_report(minimal=True, title="User uploaded table",
                                                progress_bar=True,
                                                dataset={
                                                    "description": 'This profiling report was generated by CBSL',
                                                    "copyright_holder": 'CBSL',
                                                    "copyright_year": '2022'
                                                })
                        st_profile_report(pr)
                        export = pr.to_html()
                        st.download_button(label='📥 Download Profile Report', data=export, file_name='Profile Report')


    '''
