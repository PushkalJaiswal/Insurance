from dataclasses import dataclass
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
#from streamlit.state.session_state import Value
from utilities import model_utils as m_utl, utils as utl
import pickle
from joblib import dump, load
from PATHS import MODELS


def load_view():

    st.markdown('\n')
    st.markdown('\n')
    st.title('Model Predictions ')
    

    # Add the expander to provide some information about the app
    with st.sidebar.expander("About the Page"):
        st.write("""
            This page gives the predictions and insights from predictions for the data uploaded using the model choosen in the Model building page.
            \n\n User can also clear the custom model selected and can go with the app recommneded model""")
    option = st.selectbox(
     'Choose specific domain : ',
     ('Health insurance', 'Auto insurance', 'Life insurance','General insurance'))
    # Load the data
    if option == 'Auto insurance':
        if 'carins.csv' not in os.listdir('data'):
            st.markdown("Please upload data through `Upload Data` page!")
        else:
            df = pd.read_csv('data/carins.csv')

        global data

        data = pd.read_csv('data/carins.csv')

        path_settings = os.path.join(os.getcwd(), r'settings.txt')
        with open(path_settings, 'r') as f1:
            model_selected = f1.read()

       

        if "model_selected" not in st.session_state:
            st.session_state.model_selected = False

        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False

        if "model_not_selected" not in st.session_state:
            st.session_state.model_not_selected = False

        if not model_selected or st.session_state.model_selected:
            st.session_state.model_selected = True
            pred_type = MODELS.get('RECOMMENDED_MODEL')
            st.text("No custom model found")
            st.text("Make predictions using the recommended model: " + pred_type
                    +"\n or build a model from model building page")
            predict_model(pred_type)

        elif model_selected or st.session_state.model_not_selected:
            st.session_state.model_not_selected = True
            pred_type = model_selected
            st.text("Make predictions using your selected model: "+ pred_type)
            predict_model(pred_type)
            if st.sidebar.button("Clear Model Selection",key= 'clear model selection'):
                with open(path_settings,'w') as f:
                    f.write('')
                    f.flush()
                    st.session_state.model_selected = True
                    st.session_state.model_not_selected = False
                    st.session_state.button_clicked = False


def predict_model(pred_type):
        if st.button("Predict", key=pred_type) or st.session_state.button_clicked:
            st.session_state.button_clicked = True
            if pred_type == 'Logistic regression':
                model = load(
                    open('assets/models/logisticmodel.pkl', 'rb'))
                output = m_utl.predict(model, data)
                df_predicted_combined = prepare_data_for_insights(output)
                download_option()
                display_prediction_insights(df_predicted_combined)
                display_multivariate_prediction_insights(df_predicted_combined)
            elif pred_type == 'Decision tree':
                model = load(open('assets/models/tree.pkl', "rb"))
                output = m_utl.predict(model, data)
                df_predicted_combined = prepare_data_for_insights(output)
                download_option()
                display_prediction_insights(df_predicted_combined)
                display_multivariate_prediction_insights(df_predicted_combined)
            elif pred_type == 'Random forest':
                model = load(open('assets/models/randomforestmodel.pkl', "rb"))
                output = m_utl.predict(model, data)
                df_predicted_combined = prepare_data_for_insights(output)
                download_option()
                display_prediction_insights(df_predicted_combined)
                display_multivariate_prediction_insights(df_predicted_combined)
            elif pred_type == 'Random forest 1':
                model = load(open('assets/models/randomforestmodel.pkl', "rb"))
                output = m_utl.predict(model, data)
                df_predicted_combined = prepare_data_for_insights(output)
                download_option()
                display_prediction_insights(df_predicted_combined)
                display_multivariate_prediction_insights(df_predicted_combined)
            elif pred_type == 'XG Boost':
                model = load(open('assets/models/xgbclass.pkl', 'rb'))
                output = m_utl.predict(model, data)
                df_predicted_combined = prepare_data_for_insights(output)
                download_option()
                display_prediction_insights(df_predicted_combined)
                display_multivariate_prediction_insights(df_predicted_combined)
            elif pred_type == 'Support Vector Machine':
                model = load(open('assets/models/svm.pkl', 'rb'))
                output = m_utl.predict(model, data)
                df_predicted_combined = prepare_data_for_insights(output)
                download_option()
                display_prediction_insights(df_predicted_combined)
                display_multivariate_prediction_insights(df_predicted_combined)        
            elif pred_type == 'ANN':
                model = load(open('assets/models/ANN.pkl', 'rb'))
                output = m_utl.predict(model, data)
                df_predicted_combined = prepare_data_for_insights(output)
                download_option()
                display_prediction_insights(df_predicted_combined)
                display_multivariate_prediction_insights(df_predicted_combined)
            elif pred_type == 'Ada Boost':
                model = load(open('assets/models/agbclass.pkl', 'rb'))
                output = m_utl.predict(model, data)
                df_predicted_combined = prepare_data_for_insights(output)
                download_option()
                display_prediction_insights(df_predicted_combined)
                display_multivariate_prediction_insights(df_predicted_combined)
            
        return None

data1= pd.read_csv('data/main_data1.csv')
def prepare_data_for_insights(output):
        predict_df = pd.DataFrame(output)
        df_pred_comb = data.reset_index(drop=True)
        df_predicted_combined = pd.concat([df_pred_comb, predict_df], axis=1)
        df_predicted_combined.rename(columns={0: 'prediction'}, inplace=True)
        df_predicted_combined["prediction"] = df_predicted_combined["prediction"].replace(
            [0, 1], ["No Fraud", "Fraud"])
        df_predicted_combined.to_csv('output/pred1.csv', index=False)
        st.success('The Prediction result is ready to be downloaded')
        return(df_predicted_combined)


def download_option():
        with open("output/pred1.csv", "rb") as file:
            btn = st.download_button(
                label='📥 Download Prediction Result',
                data=file,
                file_name='predictions.csv',
                mime="text/csv",
                key='download-csv'
         )
        return None


def display_prediction_insights(df_predicted_combined):
    # Actinable insights section starts here
        with st.expander("View Insights"):
            cols = pd.read_csv('data/metadata/coltypenew.csv')
            categorical, numerical = utl.getColumnTypes(cols)
            tar = "prediction"
            st.markdown('\n')

            st.subheader("Recommended features for consideration : ")
            st.markdown('\n')
            for cat in categorical:
                a = df_predicted_combined.groupby(
                    cat)["prediction"].value_counts(normalize=True).unstack()

                High_risk_cat = []
                list1 = a.index.tolist()

                for i in range(len(a)):
                    feat = list1[i]
                    if a["No Fraud"][feat] > 0.6:
                        High_risk_cat.append(list1[i])

                if len(High_risk_cat) > 0:
                    High_risk_cat_str = str(High_risk_cat)[1:-1]
                    st.write("Classes to consider in", cat, ":  ", High_risk_cat_str)
                    cat_graph_ok = st.checkbox('Display plot', key=cat)
                    if cat_graph_ok:
                        int_level = df_predicted_combined[cat].value_counts()
                        plt.xlabel("feat", size=16, )
                        plt.ylabel("Count", size=16)
                        ax = a.plot(kind='barh', stacked='True',
                                title="percentage distribution", figsize=(10, 6))
                        st.pyplot()
                        st.markdown('\n')

        # st.subheader("Numerical columns to consider")
            st.markdown('\n')
            for num in numerical:
                new = num + "_new"
                bin = 5
                df_predicted_combined[new] = pd.cut(
                    df_predicted_combined[num], bin, duplicates='drop')
                m = df_predicted_combined[new].value_counts()

                if len(m[m == 0]) > 0:
                    bin = bin - len(m[m == 0])
                    df_predicted_combined[new] = pd.cut(
                        df_predicted_combined['PolicyNumber'], bin)


                b = df_predicted_combined.groupby(new)["prediction"].value_counts(normalize=True).unstack()
                High_risk_num = []
                list2 = b.index.tolist()

                for i in range(len(b)):
                    feat = list2[i]
                    if b["No Fraud"][feat] > 0.6:
                        High_risk_num.append(list2[i])

                if len(High_risk_num) > 0:
                    High_risk_num_str = str(High_risk_num)[1:-1]
                    st.write("Numerical feature to consider ", num)
                    num_graph_ok = st.checkbox('Display plot', key=num)
                    if num_graph_ok:
                        b.plot(kind='barh', stacked='True',
                            title="percentage distribution", xlabel=num)
                        int_level = df_predicted_combined[new].value_counts()
                        st.pyplot()
                        st.markdown('\n')
            return None 


def display_multivariate_prediction_insights(df_predicted_combined):

        with st.expander("View MultiVariate Prescriptive Insights"):

            var_list = list(data.columns)

            col1, col2 = st.columns(2)

            with col1:

                var1 = st.selectbox('Select x axis feature(cat col)',var_list,key = 'var1')

            with col2:

                var2 = st.selectbox('Select y axis feature(num col)',var_list,key = 'var2')


            if st.button("plot",key='multivariate_plot'):

                plt.rcParams["figure.figsize"] = [10.00, 5.00]

                plt.rcParams["figure.autolayout"] = True



                with sns.axes_style('white'):

                    g = sns.catplot(x=var1, y=var2, hue='prediction', data=df_predicted_combined,

                        kind='bar', palette='muted')

                    st.pyplot()



                    df_5 = df_predicted_combined.sample(frac = 0.005)

                    sns.catplot(x=var1, y=var2, hue="prediction", kind="swarm", data=df_5)

                    st.pyplot()

        return None
    
    