#The app startup page
'''bla bla bla
'''

#importing the required libraries
from sklearn import model_selection
import streamlit as st
import os
from PIL import Image
import numpy as np
from utilities import utils as utl
from views import home, data, analysis, options, configuration, model_building, model_predictions, model_monitoring, explainable_AI

#setting page configuration
st.set_page_config(layout="centered", page_title='Fraud claim prediction tool')
st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()

#page banner
path_settings = os.path.join(os.getcwd(), 'assets', 'images', r'logo.png')

display = Image.open(path_settings)
display = np.array(display)

col1, col2 = st.columns(2)
col1.image(display, width = 300)
col2.title("Fraud claim prediction tool")


st.markdown(""" <style> .font1 {
            width:500px;} 
             </style> """, unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True) 

global model_selected
global option_selected
file_path_settings = os.path.join(os.getcwd(), '..\\', r'settings.txt')

#Navigation bar details
def navigation():  
    route = utl.get_current_route()
    if route == "home":
        home.load_view()
    elif route == "data":
        data.load_view()
    elif route == "analysis":
        analysis.load_view()
    elif route == "model_building":
        model_building.load_view()
    elif route == "model_predictions":
        model_predictions.load_view()
    elif route == "model_monitoring":
        model_monitoring.load_view()
    elif route == "explainable_AI":
        explainable_AI.load_view()
    elif route == "options":
        options.load_view()
    elif route == "configuration":
        configuration.load_view()
    elif route is None:
        home.load_view()
navigation()
