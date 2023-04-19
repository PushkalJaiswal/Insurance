# from ast import Num
import os
# import io
# from tkinter import *
# from soupsieve import select
import streamlit as st
#from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import  Image
#from xgboost import XGBClassifier
# from joblib import dump, load
# import base64
# from pandas.api.types import is_numeric_dtype
# from pandas_profiling import ProfileReport
# import streamlit.components.v1 as components
# import lime
# import lime.lime_tabular
# import webbrowser
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_view():
    st.write("## Claim level result explanation")
    st.write("\n")
    st.write("\n")
    with st.sidebar.expander("About the Page"):
        st.write("""
            This page gives the explanation of the model result at a userID level""")
    option = st.selectbox(
     'Choose specific domain : ',
     ('Health insurance', 'Auto insurance', 'Life insurance','General insurance'))
    if option == 'Health insurance':
    #results = pd.read_csv("output/predictions.csv")
    #results_df= pd.DataFrame(results)

    #model = load('assets/models/rf_classifier.joblib')
    #lime

    #predict_fn_rf = lambda x: model.predict_proba(x).astype(float)
        df_analysis = pd.read_csv('data/main_data.csv')
    #df_analysis =df_analysis.drop(columns='Unnamed: 0')
    #x_train = pd.read_csv("data/x_train.csv")
    #x_train =x_train.drop(columns='Unnamed: 0')
    #X = x_train.values
    #explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = x_train.columns,class_names=['True claim','False claim'],kernel_width=5)

        claim_id_list = df_analysis.index
        claim_id = st.number_input("Enter the claim id")
    #st.write(claim_id)
    #if claim_id in claim_id_list:
        #st.write("yes")
        #choosen_instance = df_analysis.loc[[claim_id]].values[0]
        #st.write(choosen_instance)
        #exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
        #exp.save_to_file("output/exp1.html")
        with open('output/exp1.html', "rb") as file:
                    st.download_button(
                    label="Download model explanation",
                    data=file,
                    file_name='model_explanation.html',
            )
    elif option == 'Auto insurance':
        df_analysis = pd.read_csv('data/fraud_oracle.csv')
    #df_analysis =df_analysis.drop(columns='Unnamed: 0')
    #x_train = pd.read_csv("data/x_train.csv")
    #x_train =x_train.drop(columns='Unnamed: 0')
    #X = x_train.values
    #explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = x_train.columns,class_names=['True claim','False claim'],kernel_width=5)

        claim_id_list = df_analysis.index
        claim_id = st.number_input("Enter the claim id")
    #st.write(claim_id)
    #if claim_id in claim_id_list:
        #st.write("yes")
        #choosen_instance = df_analysis.loc[[claim_id]].values[0]
        #st.write(choosen_instance)
        #exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
        #exp.save_to_file("output/exp1.html")
        with open('output/exp1.html', "rb") as file:
                    st.download_button(
                    label="Download model explanation",
                    data=file,
                    file_name='model_explanation.html',
            )


