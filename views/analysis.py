import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px #for visualization
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
from utilities import utils as utl
import os

def load_view():

    st.markdown('\n')
    st.markdown('\n')
    st.title('📉 Data Analysis ')

    # Add the expander to provide some information about the app
    with st.sidebar.expander("About the page"):
        st.write("""This page helps in visualizing the data distribution of the selected variables (numerical and categorical).
        A Corelation matrix for the dataset is displayed for understanding relation between variables. 
          """)
    option = st.selectbox(
     'Choose specific domain : ',
     ('Health insurance', 'Auto insurance', 'Life insurance','General insurance'))
    if option == 'Health insurance':

        if 'main_data.csv' not in os.listdir('data'):
            st.markdown("Please upload data through `Upload Data` page!")
        else:
            st.markdown("""
         <style>
         .big-markdown {
             font-size:22px;
             horizontal-align: center;
         }
         </style>
         """, unsafe_allow_html=True)

            df_analysis = pd.read_csv(r'data/main_data1.csv')
            df_visual = df_analysis.copy()
            cols = pd.read_csv(r'data/metadata/column_type_desc.csv')
            categorical, numerical = utl.getColumnTypes(cols)

            st.markdown('<p class="big-markdown"> Plotting Categorical Columns</p>', unsafe_allow_html=True)

            category = st.selectbox("Select Categorical Feature ", categorical)

            st.subheader('Value counts for selected categorical feature')
            val_count  = df_visual[category].value_counts(sort = False)
            percentage = []
            for count in val_count.values:
                p= (count/val_count.values.sum())*100
                percentage.append(p)
            
            fig = plt.figure(figsize=(10,5))
            plt.xlabel(category, size = 16,)
            plt.ylabel("Count", size = 16)
            ax= sns.barplot(x= val_count.index, y=val_count.values,order =val_count.index, saturation = 0.9)
            utl.change_width(ax, .35)

            patches = ax.patches
            for i in range(len(patches)):
                x = patches[i].get_x() + patches[i].get_width()/2
                y = patches[i].get_height()+.05
                ax.annotate('{:.1f}%'.format(percentage[i]), (x, y), ha='center')
                plt.show()

        # Add figure in streamlit app
            st.pyplot()
           
            st.markdown('\n')
            st.markdown('\n')
            st.markdown('<p class="big-markdown"> Plotting Numerical Columns</p>', unsafe_allow_html=True)
            selection = st.selectbox("Select Numerical Feature ", numerical)

            fig, ax = plt.subplots()
            ax.hist(df_visual[selection], bins=8, rwidth=0.9, color='tab:blue')
            plt.xlabel(selection, size=10, )
            plt.ylabel("Count", size=10)
        #plt.grid(axis='y', alpha=0.75)
            ax.set_title('Distribution of selected feature')
            plt.show()
            st.pyplot()

            st.markdown('\n')
            st.markdown('\n')
            st.markdown('<p class="big-markdown"> Correlation Matrix of the feature columns</p>', unsafe_allow_html=True)

            corr = df_analysis.corr(method='pearson')

            fig2, ax2 = plt.subplots()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
        # Colors
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0, ax=ax2)
            ax2.set_title("Correlation Matrix")
            st.pyplot(fig2)
    elif option == 'Auto insurance':
        if 'fraud_oracle.csv' not in os.listdir('data'):
            st.markdown("Please upload data through `Upload Data` page!")
        else:
            st.markdown("""
         <style>
         .big-markdown {
             font-size:22px;
             horizontal-align: center;
         }
         </style>
         """, unsafe_allow_html=True)

            df_analysis = pd.read_csv(r'data/fraud_oracle.csv')
            df_visual = df_analysis.copy()
            cols = pd.read_csv(r'data/metadata/coltypecsv.csv')
            categorical, numerical = utl.getColumnTypes(cols)

            st.markdown('<p class="big-markdown"> Plotting Categorical Columns</p>', unsafe_allow_html=True)

            category = st.selectbox("Select Categorical Feature ", categorical)

            st.subheader('Value counts for selected categorical feature')
            val_count  = df_visual[category].value_counts(sort = False)
            percentage = []
            for count in val_count.values:
                p= (count/val_count.values.sum())*100
                percentage.append(p)
            
            fig = plt.figure(figsize=(10,5))
            plt.xlabel(category, size = 16,)
            plt.ylabel("Count", size = 16)
            ax= sns.barplot(x=val_count.index, y=val_count.values,order =val_count.index, saturation = 0.9)
            utl.change_width(ax, .35)

            patches = ax.patches
            for i in range(len(patches)):
                x = patches[i].get_x() + patches[i].get_width()/2
                y = patches[i].get_height()+.05
                ax.annotate('{:.1f}%'.format(percentage[i]), (x, y), ha='center')
                plt.show()

        # Add figure in streamlit app
            st.pyplot()
           
            st.markdown('\n')
            st.markdown('\n')
            st.markdown('<p class="big-markdown"> Plotting Numerical Columns</p>', unsafe_allow_html=True)
            selection = st.selectbox("Select Numerical Feature ", numerical)

            fig, ax = plt.subplots()
            ax.hist(df_visual[selection], bins=8, rwidth=0.9, color='tab:blue')
            plt.xlabel(selection, size=10, )
            plt.ylabel("Count", size=10)
        #plt.grid(axis='y', alpha=0.75)
            ax.set_title('Distribution of selected feature')
            plt.show()
            st.pyplot()

            st.markdown('\n')
            st.markdown('\n')
            st.markdown('<p class="big-markdown"> Correlation Matrix of the feature columns</p>', unsafe_allow_html=True)

            corr = df_analysis.corr(method='pearson')

            fig2, ax2 = plt.subplots()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
        # Colors
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0, ax=ax2)
            ax2.set_title("Correlation Matrix")
            st.pyplot(fig2)

