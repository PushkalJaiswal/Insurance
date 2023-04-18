import streamlit as st


def load_view():
    st.title('Options Page')

   # Add the expander to provide some information about the app
    with st.sidebar.expander("About the App"):
        st.write("""
            This data profiling App was built by Sharone Li using Streamlit and pandas_profiling package. You can use the app to quickly generate a comprehensive data profiling and EDA report without the need to write any python code. \n\nThe app has the minimum mode (recommended) and the complete code. The complete code includes more sophisticated analysis such as correlation analysis or interactions between variables which may requires expensive computations. )
         """)