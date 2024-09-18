import streamlit as st


def about_app():
    st.subheader('About')
    with st.expander('Supported Machine Learning Models'):
        st.markdown(''' 
        * Logistic Regression 
        * Support Vector Machine
        * Decision Tree 
        * Random Forest
        * Gaussian NB
        ''')
    with st.expander('Technologies Used'):
        st.markdown(''' 
        * numpy -- Numerical operations
        * pandas -- Data manipulation and analysis
        * scikit_learn -- For machine learning
        * streamlit -- Front end
        ''')
    with st.expander('Data Source'):
        st.markdown(''' 
        * [Diabetes Disease](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
        * [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
        * [Parkinson Disease](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)
        * [Liver Disease](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records/data)
        * [Kidney Disease](https://www.kaggle.com/datasets/mansoordaku/ckdisease/data)
        ''')
    with st.expander('Contact'):
        st.markdown(''' Any Queries: Contact [Zeeshan Altaf](mailto:zeeshan.altaf@92labs.ai)''')
    with st.expander('Source Code'):
        st.markdown(''' Source code: [GitHub](https://github.com/mzeeshanaltaf/)''')


st.title('MLxplorer 🚀')
st.write(':blue[***Unleash the Power of Predictive Insights⚡***]')
st.write("MLxplorer is a dynamic web platform that brings together multiple mini-apps 📊, each designed to analyze "
         "diverse datasets using advanced machine learning techniques 🤖. Whether you're predicting outcomes, "
         "exploring patterns 🔍, or uncovering hidden insights 💡, MLxplorer equips you with the tools to harness "
         "the power of AI across various domains—all in one place!")

about_app()
