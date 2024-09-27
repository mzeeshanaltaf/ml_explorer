import streamlit as st
from util import *
from models import *


# Function to display the model prediction and probability

def display_prediction(pred, prob):
    st.subheader("Prediction")
    df_prob = pd.DataFrame(prob)
    df_prob.columns = ['No Heart Disease', 'Heart Disease']

    if pred[0] == 0:
        prob = "{:.3f}".format(prob[0][0])
        st.success(f"Heart Disease: NO")
    else:
        prob = "{:.3f}".format(prob[0][0])
        st.error(f"Heart Disease: YES")

    st.subheader('Prediction Probability')
    st.dataframe(df_prob,
                 column_config={
                     'No Heart Disease': st.column_config.ProgressColumn(
                         'No Heart Disease',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Heart Disease': st.column_config.ProgressColumn(
                         'Heart Disease',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                 }, hide_index=True)


# Application title and description
st.title('HeartCheck ❤️')
st.write(":blue[***Your Heart's Health, One Click Away 💓***]")
st.write(
    "HeartCheck is a web app that empowers users to assess their heart health 💖. By inputting key health metrics, "
    "the app uses advanced machine learning algorithms 🤖 to predict the likelihood of heart disease. Stay ahead "
    "of your health with quick, reliable insights and take control of your well-being 🩺!")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).', icon='ℹ️')

# Show Disclaimer
with st.expander('Disclaimer', icon=':material/info:'):
    display_disclaimer()

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = heart_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Heart', 'binary')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# Display Confusion Matrix
labels = ['Disease', 'Not Disease']
display_confusion_matrix(cm, labels)
