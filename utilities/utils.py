import streamlit as st
import base64
import numpy as np
from streamlit.components.v1 import html
from pandas.api.types import is_numeric_dtype
import pandas as pd
import os
import pickle
from PATHS import NAVBAR_PATHS, SETTINGS
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, f1_score
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split


def inject_custom_css():
    path = os.path.join(os.getcwd(), 'assets', r'styles.css')
    with open(path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def getColumnTypes(cols):
    categorical=[]
    numerical = []
    #Object = []
    for i in range(len(cols)):
        if cols["type"][i]=='categorical':
            categorical.append(cols['column_name'][i])
        elif cols["type"][i]=='numerical':
            numerical.append(cols['column_name'][i])
        #else:
            #Object.append(cols['column_name'][i])
    return categorical, numerical #Object


def isCategorical(col):
    unis = np.unique(col)
    if len(unis) < 0.02*len(col):
        return True
    return False

def isNumerical(col):
    return is_numeric_dtype(col)


def get_current_route():
    try:
        return st.experimental_get_query_params()['nav'][0]
    except:
        return None


def genMetaData(df):
    col = df.columns
    ColumnType = []
    Object = []
    Numerical = df.select_dtypes(include=np.number).columns.tolist()
    Categorical = list(set(list(df.columns)) - set(Numerical))

    for i in (col):
        if i in Numerical:
            ColumnType.append((i, "numerical"))
        elif i in Categorical:
            ColumnType.append((i, "categorical"))
        else:
            ColumnType.append((i, "object"))
            Object.append(i)
    return ColumnType


def labelEncodeData(encodedData):
    file = open("assets/models/lbl_enc.obj", 'rb')
    lbl_enc = pickle.load(file)
    file.close()
    encodedData['job'] = lbl_enc.fit_transform(encodedData.job.values)
    encodedData['marital'] = lbl_enc.fit_transform(encodedData.marital.values)
    encodedData['education'] = lbl_enc.fit_transform(encodedData.education.values)
    encodedData['default'] = lbl_enc.fit_transform(encodedData.default.values)
    encodedData['housing'] = lbl_enc.fit_transform(encodedData.housing.values)
    encodedData['loan'] = lbl_enc.fit_transform(encodedData.loan.values)
    encodedData['contact'] = lbl_enc.fit_transform(encodedData.contact.values)
    encodedData['month'] = lbl_enc.fit_transform(encodedData.month.values)
    encodedData['day_of_week'] = lbl_enc.fit_transform(encodedData.day_of_week.values)
    encodedData['poutcome'] = lbl_enc.fit_transform(encodedData.poutcome.values)
    return encodedData


def decodeDataWithInversetransform(decoded):
    file = open("assets/models/lbl_enc.obj", 'rb')
    lbl_enc = pickle.load(file)
    file.close()
    decoded['job'] = lbl_enc.inverse_transform(decoded.job.values)
    decoded['marital'] = lbl_enc.inverse_transform(decoded.marital.values)
    decoded['education'] = lbl_enc.inverse_transform(decoded.education.values)
    decoded['default'] = lbl_enc.inverse_transform(decoded.default.values)
    decoded['housing'] = lbl_enc.inverse_transform(decoded.housing.values)
    decoded['loan'] = lbl_enc.inverse_transform(decoded.loan.values)
    decoded['contact'] = lbl_enc.inverse_transform(decoded.contact.values)
    decoded['month'] = lbl_enc.inverse_transform(decoded.month.values)
    decoded['day_of_week'] = lbl_enc.inverse_transform(decoded.day_of_week.values)
    decoded['poutcome'] = lbl_enc.inverse_transform(decoded.poutcome.values)
    return decoded


def normaliseData(scaledData):
    file = open("assets/models/min_max_scaler.obj", 'rb')
    min_max_scaler = pickle.load(file)
    file.close()
    scaledData[['pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
        'nr.employed']] = min_max_scaler.fit_transform(
        scaledData[['pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']])
    return scaledData


def preprocessData(df):
    df = df.drop(['duration', 'id', 'y'], 1)
    df = labelEncodeData(df)
    df = normaliseData(df)
    return df


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def makeMapDict(col):
    uniqueVals = list(np.unique(col))
    uniqueVals.sort()
    dict_ = {uniqueVals[i]: i for i in range(len(uniqueVals))}
    return dict_


def mapunique(df, colName):
    dict_ = makeMapDict(df[colName])
    cat = np.unique(df[colName])
    df[colName] = df[colName].map(dict_)
    return cat


# For redundant columns
def getRedundentColumns(corr, y: str, threshold=0.1):
    cols = corr.columns
    redunt = []
    k = 0
    for ind, c in enumerate(corr[y]):
        if c < 1 - threshold:
            redunt.append(cols[ind])
    return redunt


def navbar_component():
    path_settings = os.path.join(os.getcwd(), 'assets','images', r'settings.png')
    with open(path_settings, "rb") as image_file:
        image_as_base64 = base64.b64encode(image_file.read())

    navbar_items = ''
    for key, value in NAVBAR_PATHS.items():
        navbar_items += (f'<a class="navitem" href="/?nav={value}">{key}</a>')

    settings_items = ''
    for key, value in SETTINGS.items():
        settings_items += (
            f'<a href="/?nav={value}" class="settingsNav">{key}</a>')

    component = rf'''
            <nav class="container navbar" id="navbar">
                <ul class="navlist">
                {navbar_items}
                </ul>
                <div class="dropdown" id="settingsDropDown">
                    <img class="dropbtn" src="data:image/png;base64, {image_as_base64.decode("utf-8")}"/>
                    <div id="myDropdown" class="dropdown-content">
                        {settings_items}
                    </div>
                </div>
            </nav>
            '''
    st.markdown(component, unsafe_allow_html=True)
    js = '''
    <script>
        // navbar elements
        var navigationTabs = window.parent.document.getElementsByClassName("navitem");
        var cleanNavbar = function(navigation_element) {
            navigation_element.removeAttribute('target')
        }

        for (var i = 0; i < navigationTabs.length; i++) {
            cleanNavbar(navigationTabs[i]);
        }

        // Dropdown hide / show
        var dropdown = window.parent.document.getElementById("settingsDropDown");
        dropdown.onclick = function() {
            var dropWindow = window.parent.document.getElementById("myDropdown");
            if (dropWindow.style.visibility == "hidden"){
                dropWindow.style.visibility = "visible";
            }else{
                dropWindow.style.visibility = "hidden";
            }
        };

        var settingsNavs = window.parent.document.getElementsByClassName("settingsNav");
        var cleanSettings = function(navigation_element) {
            navigation_element.removeAttribute('target')
        }

        for (var i = 0; i < settingsNavs.length; i++) {
            cleanSettings(settingsNavs[i]);
        }
    </script>
    '''
    html(js)



def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


#not used yet
def plot_decision_boundary_and_metrics(model, x_train, y_train, x_test, y_test, metrics):
    d = x_train.shape[1]

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    model_input = [(xx.ravel() ** p, yy.ravel() ** p) for p in range(1, d // 2 + 1)]
    aux = []
    for c in model_input:
        aux.append(c[0])
        aux.append(c[1])

    Z = model.predict(np.concatenate([v.reshape(-1, 1) for v in aux], axis=1))

    Z = Z.reshape(xx.shape)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Decision Boundary", None, None),
        row_heights=[0.7, 0.30],
    )

    heatmap = go.Heatmap(
        x=xx[0],
        y=y_,
        z=Z,
        colorscale=["tomato", "rgb(27,158,119)"],
        showscale=False,
    )

    train_data = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        name="train data",
        mode="markers",
        showlegend=True,
        marker=dict(
            size=10,
            color=y_train,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    test_data = go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        name="test data",
        mode="markers",
        showlegend=True,
        marker_symbol="cross",
        visible="legendonly",
        marker=dict(
            size=10,
            color=y_test,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(heatmap, row=1, col=1,).add_trace(train_data).add_trace(
        test_data
    ).update_xaxes(range=[x_min, x_max], title="x1").update_yaxes(
        range=[y_min, y_max], title="x2"
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_accuracy"],
            title={"text": f"Accuracy (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_accuracy"]},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_f1"],
            title={"text": f"F1 score (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_f1"]},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=700,
    )

    return fig


def insert_string_middle(str, word,position):
    return str[:position] + word + str[position:]


def generate_data(data,test_size):
    x= data.drop('PotentialFraud', axis=1)
    y= data.loc[:,'PotentialFraud']
    x_train, y_train, x_test, y_test = train_test_split(x,y, test_size=test_size)
    return x_train, y_train, x_test,y_test

def generate_data_auto(data,test_size):
    x=data.drop('FraudFound_P',axis=1)
    y=data.loc[:,'FraudFound_P']
    x_train, y_train, x_test, y_test = train_test_split(x,y, test_size=test_size)
    return x_train, y_train, x_test,y_test