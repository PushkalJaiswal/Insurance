import streamlit as st
import os
from PIL import Image
import numpy as np

def load_view():
    # st.title('Bank Telemarketing outcome prediction')
    # with open(path_settings, "rb") as image_file:

    st.write("\n")
    st.markdown('<h5>Description:</h5>', unsafe_allow_html=True) 
    st.markdown("""This tool helps to predict the potentially fraudulent claims based on the claims filed in the past. It also helps in discovering important variables helpful in detecting the behaviour of potentially fraud providers""")
    st.write("\n")
    st.write("\n")
    
    option_selected = st.selectbox(
     'Choose specific domain : ',
     ('Health insurance', 'Auto insurance', 'Life insurance','General insurance'))

    
    #The expander to provide some information about the app
    with st.sidebar.expander("About the App"):
        st.write("""
            This App was built  using Streamlit. You can use the app to quickly generate a comprehensive data profiling, EDA report and claim level predictions based on the historical data 
            without the need to write any python code. """)
        


