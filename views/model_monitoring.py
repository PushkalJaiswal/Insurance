import streamlit as st
import os
from PIL import Image
import numpy as np
import streamlit.components.v1 as components



def load_view():

    st.markdown('\n')
    st.markdown('\n')
    st.title('Model Predictions ')
    with st.sidebar.expander("About the Page"):
        st.write("""
            This page captures the model accuracy of last 6 months, which helps in model monitoring for model retraining purpose""")
            
        st.header('Performance evaluation')
    option = st.selectbox(
     'Choose specific domain : ',
     ('Health insurance', 'Auto insurance', 'Life insurance','General insurance'))
    if option == 'Health insurance':    
        path_settings = os.path.join(os.getcwd(), 'assets', 'images', r'Picture1.png')
        image1 = Image.open(path_settings)
        image1 = np.array(image1)
        st.image(image1)
        st.write('The Accuracy of the Predictions made on last month''s data is 88.4%')
        st.write('The Accuracy of the Predictions made on this month''s data is 89.2%')
        path_settings = os.path.join(os.getcwd(), r'exp1.html')

        with open(path_settings, 'r', encoding='utf-8') as HtmlFile:
            source_code = HtmlFile.read()
        #st.write(source_code) 
    #print(source_code)
            components.html(source_code)

        st.markdown('\n')
        st.header('Error diagnosis')

    # Add the expander to provide some information about the app
    
    elif option == 'Auto insurance':
        st.header('Performance evaluation')
        path_settings = os.path.join(os.getcwd(), 'assets', 'images', r'Picture1.png')
        image1 = Image.open(path_settings)
        image1 = np.array(image1)
        st.image(image1)
        st.write('The Accuracy of the Predictions made on last month''s data is 88.4%')
        st.write('The Accuracy of the Predictions made on this month''s data is 89.2%')
        path_settings = os.path.join(os.getcwd(), r'exp1.html')

        with open(path_settings, 'r', encoding='utf-8') as HtmlFile:
            source_code = HtmlFile.read()
        #st.write(source_code) 
    #print(source_code)
            components.html(source_code)

        st.markdown('\n')
        st.header('Error diagnosis') 
        


