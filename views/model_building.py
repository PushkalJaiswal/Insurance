from glob import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#from PIL import Image
from utilities import model_utils as m_utl, utils as utl
import pickle
from joblib import dump, load
import time
from sklearn.metrics import accuracy_score, f1_score

def load_view():

    global model_selected
    global data
    
    st.markdown('\n')
    st.markdown('\n')
    st.title('⚙️Model building')

    # Add the expander to provide some information about the app
    with st.sidebar.expander("About the page"):
        st.write("""
            User can custom build the model by changing the hyper-parameters using this page. """)
    option = st.selectbox(
     'Choose specific domain : ',
     ('Health insurance', 'Auto insurance', 'Life insurance','General insurance'))
    if option == 'Health insurance':
    # Load the data
        if 'main_data1.csv' not in os.listdir('data'):
            st.markdown("Please upload data through `Upload Data` page!")
        else:
            df = pd.read_csv('data/main_data1.csv')
            data = df 

        if "model_selected_to_build" not in st.session_state:
            st.session_state.model_selected_to_build = False


        model_selected = st.selectbox("Select the machine learning model",("Logistic regression","Decision tree","Random forest","XG Boost", "Support Vector Machine", "ANN"))
        path_settings = os.path.join(os.getcwd(), r'settings.txt')
        st.session_state['app_mode'] = model_selected

        sidebar_expander = st.sidebar.expander("Train validation set creation")
        with sidebar_expander:    
            test_size = st.slider("test size ", 0, 1, value=(0,1))  
    
        sidebar_expander_fs = st.sidebar.expander("Feature Selection")
        df = pd.read_csv('data/main_data1.csv')
        data = df
        with sidebar_expander_fs:  
            var_list = list(data.columns)
            option3 = st.multiselect('Select Features',var_list)
            data = data[option3]
        st.sidebar.markdown("""---""")

        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False

        if st.button("select model template") or st.session_state.button_clicked:
            st.session_state.button_clicked = True
            with open(path_settings,'w') as f:
                f.write(model_selected)
                f.flush()

            if model_selected == 'Logistic regression':
                sidebar_expander_lg = st.sidebar.expander("Configure Logistic Regression",True)
                with sidebar_expander_lg: 
                    tooltip_text_lg_selectbox = "suggested solver is newton-cg"
                    solver = st.selectbox("solver", options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"],help=tooltip_text_lg_selectbox)
                    if solver in ["newton-cg", "lbfgs", "sag"]:
                        tooltip_text_lg_penalty_selectbox = "suggested penalty is l2"
                        penalties = ["l2", "none"]
                    elif solver == "saga":
                        tooltip_text_lg_penalty_selectbox = "suggested penalty is elasticnet"
                        penalties = ["l1", "l2", "none", "elasticnet"]
                    elif solver == "liblinear":
                        tooltip_text_lg_penalty_selectbox = "suggested penalty is l1"
                        penalties = ["l1"]
                    penalty = st.selectbox("penalty", options=penalties, help = tooltip_text_lg_penalty_selectbox)
                    tooltip_text_lg_iter_selectbox = "suggested penalty is 10"
                    max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100, help = tooltip_text_lg_iter_selectbox)
                    params = {"solver": solver, "penalty": penalty, "max_iter": max_iter}
                if st.button('Build',key='build_lr'):
                #train_model(model,x_train,y_train,x_test,y_test)
                    st.success('Logistic regression model is built and ready for download')
                    with open('assets/models/logistic_regression_model.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='logistic_regression_model.pkl',
                ) 

            elif model_selected == "Decision tree" :
                sidebar_expander_dt = st.sidebar.expander("Configure Decision Tree",True)
                with sidebar_expander_dt:  
                    st.write("Build ParamGrid")
                    tooltip_text_dt_selectbox = "suggested criterion is gini"
                    criterion = st.selectbox("criterion", ["gini", "entropy"],help = tooltip_text_dt_selectbox)
                    tooltip_text_dt_selectbox2 = "suggested depth is 7"
                    max_depth = st.number_input("max_depth", 1, 50, 5, 1, help = tooltip_text_dt_selectbox2)
                    tooltip_text_dt_selectbox3 = "suggested max_features auto"
                    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"], help = tooltip_text_dt_selectbox3)
            
                if st.button('Build',key='build_dt'):
                    st.success('Decision tree model is built and ready for download')
                    with open('assets/models/decision_tree_model.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='decision_tree_model.pkl',
                )

            elif model_selected == "Random forest" :
                sidebar_expander_rf = st.sidebar.expander("Configure Random Forest",True)
                with sidebar_expander_rf: 
                    tooltip_text_rf_selectbox = "suggested criterion is gini"
                    criterion = st.selectbox("criterion", ["gini", "entropy"],help = tooltip_text_rf_selectbox)
                    tooltip_text_rf_estimators = "suggested estimators is 50"
                    n_estimators = st.number_input("n_estimators", 50, 300, 100, 10, help = tooltip_text_rf_estimators)
                    tooltip_text_rf_selectbox2 = "suggested depth is 7"
                    max_depth = st.number_input("max_depth", 1, 50, 5, 1, help= tooltip_text_rf_selectbox2)
                    tooltip_text_rf_selectbox3 = "suggested max_features auto"
                    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"], help = tooltip_text_rf_selectbox3)

                params = {
                    "criterion": criterion,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "n_jobs": -1,
            }

                if st.button('Build',key='build_rf'):
                    st.success('Random forest model is built and ready for download')
                    with open('assets/models/random_forest_model.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='random_forest_model.pkl',
                )

            elif model_selected == "XG Boost" :
                sidebar_expander_xg = st.sidebar.expander("Configure XG Boost",True)
                with sidebar_expander_xg: 
                    tooltip_text_xg_selectbox = "suggested criterion is gini"
                    criterion = st.selectbox("criterion", ["gini", "entropy"],help = tooltip_text_xg_selectbox)
                    tooltip_text_xg_estimators = "suggested estimators is 50"
                    n_estimators = st.number_input("n_estimators", 50, 300, 100, 10, help = tooltip_text_xg_estimators)
                    tooltip_text_xg_selectbox2 = "suggested depth is 7"
                    max_depth = st.number_input("max_depth", 1, 50, 5, 1, help= tooltip_text_xg_selectbox2)
                    tooltip_text_xg_selectbox3 = "suggested max_features auto"
                    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"], help = tooltip_text_xg_selectbox3)

                params = {
                    "criterion": criterion,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "n_jobs": -1,
            }

                if st.button('Build',key='build_rf'):
                    st.success('XG Boost model is built and ready for download')
                    with open('assets/models/xgb_model.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='xgb_model.pkl',
                )

            
            elif model_selected == "Support Vector Machine" :
                sidebar_expander_xg = st.sidebar.expander("Configure SVM",True)
                with sidebar_expander_xg: 
                    tooltip_text_svm_penalty = "suggested Penalty is gini"
                    penalty = st.number_input("Penalty for error term", 0.1, 1.0, 0.50, 0.1, help = tooltip_text_svm_penalty)
                    tooltip_text_svm_gamma = "suggested gamma value is 0.1"
                    gamma = st.number_input("Gamma",  0.1, 1.0, 0.1, 0.1, help = tooltip_text_svm_gamma)
                    tooltip_text_svm_Kernel = "suggested kernel for our usecase is sigmoid"
                    kernel = st.selectbox("Kernel", ['rbf', 'poly', 'sigmoid', 'linear'], help= tooltip_text_svm_Kernel)

                params = {
                    "penalty": penalty,
                    "gamma": gamma,
                    "kernel": kernel
            }

                if st.button('Build',key='build_svm'):
                    st.success('SVM Boost model is built and ready for download')
                    with open('assets/models/svm.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='svm.pkl',
                )


            elif model_selected == "ANN" :
                sidebar_expander_xg = st.sidebar.expander("Configure ANN",True)
                with sidebar_expander_xg: 
                    tooltip_text_ANN_1 = "suggested input layer activation function is None"
                    input_act_func = st.selectbox("Input layer activation function", ["None", "Sigmoid", "ReLu"],help = tooltip_text_ANN_1)
                    tooltip_text_ANN_2 = "suggested number of hidden layers 4"
                    N_hidden = st.number_input("Number of hidden layers", 1, 100, 5, 1, help = tooltip_text_ANN_2)
                    tooltip_text_ANN_3 = "suggested dropout layer position is 3rd"
                    dropout_units = st.number_input("Dropout layer position", 1, 100, 3, 1, help = tooltip_text_ANN_3)
                    tooltip_text_ANN_3 = "suggested output layer activation function is sigmoid"
                    output_act_func = st.selectbox("Output layer activation function", ['sigmoid', 'softmax'], help= tooltip_text_ANN_3)
                    tooltip_text_ANN_4 = "suggested Optimizer is adam"
                    optimizer_func = st.selectbox("Optimizer", ['Gradient Descent', 'stochastic gradient descent', 'adam', 'adagrad','RMSprop'], help= tooltip_text_ANN_3)
                    tooltip_text_ANN_5 = "suggested loss function is binary crossentropy"
                    loss_func = st.selectbox("Loss function", ['Regression Loss Function','Mean Squared Error','binary crossentropy',
                    'Mean Squared Logarithmic Error Loss','Mean Absolute Error Loss','Binary Classification Loss Function'], help= tooltip_text_ANN_3)
        
                if st.button('Build',key='build_ann'):
                    st.success('ANN Boost model is built and ready for download')
                    with open('assets/models/ANN.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='ANN.pkl',
                )
    elif option == 'Auto insurance':
        if 'main_data1.csv' not in os.listdir('data'):
            st.markdown("Please upload data through `Upload Data` page!")
        else:
            df = pd.read_csv('data/main_data1.csv')
            data = df 

        if "model_selected_to_build" not in st.session_state:
            st.session_state.model_selected_to_build = False


        model_selected = st.selectbox("Select the machine learning model",("Logistic regression","Decision tree","Random forest","XG Boost", "Ada Boost"))
        path_settings = os.path.join(os.getcwd(), r'settings.txt')
        st.session_state['app_mode'] = model_selected

        sidebar_expander = st.sidebar.expander("Train validation set creation")
        with sidebar_expander:    
            test_size = st.slider("test size ", 0, 1, value=(0,1))  
    
        sidebar_expander_fs = st.sidebar.expander("Feature Selection")
        df = pd.read_csv('data/main_data1.csv')
        data = df
        with sidebar_expander_fs:  
            var_list = list(data.columns)
            option3 = st.multiselect('Select Features',var_list)
            data = data[option3]
        st.sidebar.markdown("""---""")

        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False

        if st.button("select model template") or st.session_state.button_clicked:
            st.session_state.button_clicked = True
            with open(path_settings,'w') as f:
                f.write(model_selected)
                f.flush()

            if model_selected == 'Logistic regression':
                sidebar_expander_lg = st.sidebar.expander("Configure Logistic Regression",True)
                with sidebar_expander_lg: 
                    tooltip_text_lg_selectbox = "suggested solver is newton-cg"
                    solver = st.selectbox("solver", options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"],help=tooltip_text_lg_selectbox)
                    if solver in ["newton-cg", "lbfgs", "sag"]:
                        tooltip_text_lg_penalty_selectbox = "suggested penalty is l2"
                        penalties = ["l2", "none"]
                    elif solver == "saga":
                        tooltip_text_lg_penalty_selectbox = "suggested penalty is elasticnet"
                        penalties = ["l1", "l2", "none", "elasticnet"]
                    elif solver == "liblinear":
                        tooltip_text_lg_penalty_selectbox = "suggested penalty is l1"
                        penalties = ["l1"]
                    penalty = st.selectbox("penalty", options=penalties, help = tooltip_text_lg_penalty_selectbox)
                    tooltip_text_lg_iter_selectbox = "suggested penalty is 10"
                    max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100, help = tooltip_text_lg_iter_selectbox)
                    params = {"solver": solver, "penalty": penalty, "max_iter": max_iter}
                if st.button('Build',key='build_lr'):
                #train_model(model,x_train,y_train,x_test,y_test)
                    st.success('Logistic regression model is built and ready for download')
                    with open('assets/models/lrmodel.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='logistic_regression_model.pkl',
                ) 

            elif model_selected == "Decision tree" :
                sidebar_expander_dt = st.sidebar.expander("Configure Decision Tree",True)
                with sidebar_expander_dt:  
                    st.write("Build ParamGrid")
                    tooltip_text_dt_selectbox = "suggested criterion is gini"
                    criterion = st.selectbox("criterion", ["gini", "entropy"],help = tooltip_text_dt_selectbox)
                    tooltip_text_dt_selectbox2 = "suggested depth is 7"
                    max_depth = st.number_input("max_depth", 1, 50, 5, 1, help = tooltip_text_dt_selectbox2)
                    tooltip_text_dt_selectbox3 = "suggested max_features auto"
                    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"], help = tooltip_text_dt_selectbox3)
            
                if st.button('Build',key='build_dt'):
                    st.success('Decision tree model is built and ready for download')
                    with open('assets/models/decntree.pkl',"rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='decision_tree_model.pkl',
                )

            elif model_selected == "Random forest" :
                sidebar_expander_rf = st.sidebar.expander("Configure Random Forest",True)
                with sidebar_expander_rf: 
                    tooltip_text_rf_selectbox = "suggested criterion is gini"
                    criterion = st.selectbox("criterion", ["gini", "entropy"],help = tooltip_text_rf_selectbox)
                    tooltip_text_rf_estimators = "suggested estimators is 50"
                    n_estimators = st.number_input("n_estimators", 50, 300, 100, 10, help = tooltip_text_rf_estimators)
                    tooltip_text_rf_selectbox2 = "suggested depth is 7"
                    max_depth = st.number_input("max_depth", 1, 50, 5, 1, help= tooltip_text_rf_selectbox2)
                    tooltip_text_rf_selectbox3 = "suggested max_features auto"
                    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"], help = tooltip_text_rf_selectbox3)

                params = {
                    "criterion": criterion,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "n_jobs": -1,
            }

                if st.button('Build',key='build_rf'):
                    st.success('Random forest model is built and ready for download')
                    with open('assets/models/rf.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='random_forest_model.pkl',
                )

            elif model_selected == "XG Boost" :
                sidebar_expander_xg = st.sidebar.expander("Configure XG Boost",True)
                with sidebar_expander_xg: 
                    tooltip_text_xg_selectbox = "suggested criterion is gini"
                    criterion = st.selectbox("criterion", ["gini", "entropy"],help = tooltip_text_xg_selectbox)
                    tooltip_text_xg_estimators = "suggested estimators is 50"
                    n_estimators = st.number_input("n_estimators", 50, 300, 100, 10, help = tooltip_text_xg_estimators)
                    tooltip_text_xg_selectbox2 = "suggested depth is 7"
                    max_depth = st.number_input("max_depth", 1, 50, 5, 1, help= tooltip_text_xg_selectbox2)
                    tooltip_text_xg_selectbox3 = "suggested max_features auto"
                    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"], help = tooltip_text_xg_selectbox3)

                params = {
                    "criterion": criterion,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "n_jobs": -1,
                }
                if st.button('Build',key='build_rf'):
                    st.success('XG Boost model is built and ready for download')
                    with open('assets/models/xgboost.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='xgb_model.pkl',
                )
    
            
            elif model_selected == "Ada Boost" :
                sidebar_expander_ada = st.sidebar.expander("Configure Ada Boost",True)
                with sidebar_expander_ada: 
                    tooltip_text_ada_estimators = "suggested estimators is 100"
                    n_estimators = st.number_input("n_estimators", 50, 300, 100, 10, help = tooltip_text_ada_estimators)
                    #tooltip_text_ada_selectbox2 = "suggested base estimator is none"
                    #base_estimator = st.number_input("base_estimator",0, help= tooltip_text_ada_selectbox2)
                    #tooltip_text_ada_selectbox3 = "suggested learning_rate"
                    #learning_rate = st.selectbox("learning_rate",1, help = tooltip_text_ada_selectbox3)
                    #tooltip_text_ada_selectbox4 = "Random state is 1"
                    #random_state = st.selectbox("Random state",1,10,100 , help = tooltip_text_ada_selectbox4)

                params = {
                    "n_estimators": n_estimators,
                    #"base_estimator": base_estimator,
                    #"learning_rate": learning_rate,
                    #"random_state": random_state,
                    
            }

                if st.button('Build',key='build_rf'):
                    st.success('Ada Boost model is built and ready for download')
                    with open('assets/models/agbclass.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='agb_model.pkl',
                )

            '''elif model_selected == "Support Vector Machine" :
                sidebar_expander_xg = st.sidebar.expander("Configure SVM",True)
                with sidebar_expander_xg: 
                    tooltip_text_svm_penalty = "suggested Penalty is gini"
                    penalty = st.number_input("Penalty for error term", 0.1, 1.0, 0.50, 0.1, help = tooltip_text_svm_penalty)
                    tooltip_text_svm_gamma = "suggested gamma value is 0.1"
                    gamma = st.number_input("Gamma",  0.1, 1.0, 0.1, 0.1, help = tooltip_text_svm_gamma)
                    tooltip_text_svm_Kernel = "suggested kernel for our usecase is sigmoid"
                    kernel = st.selectbox("Kernel", ['rbf', 'poly', 'sigmoid', 'linear'], help= tooltip_text_svm_Kernel)

                params = {
                    "penalty": penalty,
                    "gamma": gamma,
                    "kernel": kernel
            }

                if st.button('Build',key='build_svm'):
                    st.success('SVM Boost model is built and ready for download')
                    with open('assets/models/svm.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='svm.pkl',
                )


            elif model_selected == "ANN" :
                sidebar_expander_xg = st.sidebar.expander("Configure ANN",True)
                with sidebar_expander_xg: 
                    tooltip_text_ANN_1 = "suggested input layer activation function is None"
                    input_act_func = st.selectbox("Input layer activation function", ["None", "Sigmoid", "ReLu"],help = tooltip_text_ANN_1)
                    tooltip_text_ANN_2 = "suggested number of hidden layers 4"
                    N_hidden = st.number_input("Number of hidden layers", 1, 100, 5, 1, help = tooltip_text_ANN_2)
                    tooltip_text_ANN_3 = "suggested dropout layer position is 3rd"
                    dropout_units = st.number_input("Dropout layer position", 1, 100, 3, 1, help = tooltip_text_ANN_3)
                    tooltip_text_ANN_3 = "suggested output layer activation function is sigmoid"
                    output_act_func = st.selectbox("Output layer activation function", ['sigmoid', 'softmax'], help= tooltip_text_ANN_3)
                    tooltip_text_ANN_4 = "suggested Optimizer is adam"
                    optimizer_func = st.selectbox("Optimizer", ['Gradient Descent', 'stochastic gradient descent', 'adam', 'adagrad','RMSprop'], help= tooltip_text_ANN_3)
                    tooltip_text_ANN_5 = "suggested loss function is binary crossentropy"
                    loss_func = st.selectbox("Loss function", ['Regression Loss Function','Mean Squared Error','binary crossentropy',
                    'Mean Squared Logarithmic Error Loss','Mean Absolute Error Loss','Binary Classification Loss Function'], help= tooltip_text_ANN_3)
        
                if st.button('Build',key='build_ann'):
                    st.success('ANN Boost model is built and ready for download')
                    with open('assets/models/ANN.pkl', "rb") as file:
                        st.download_button(
                        label="Download model",
                        data=file,
                        file_name='ANN.pkl',
                )

    ''''''(model,
        train_accuracy,
        train_f1,
        test_accuracy,
        test_f1,
        duration,
    ) =train_model(model, x_train, y_train, x_test, y_test)

    metrics = {
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
    }

    fig = utl.plot_decision_boundary_and_metrics(
        model_selected, x_train, y_train, x_test, y_test, metrics
    )

def train_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)

    return model, train_accuracy, train_f1, test_accuracy, test_f1, duration

'''

