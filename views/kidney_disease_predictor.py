import streamlit as st
import pandas as pd
from util import *
from models import *


# Function to display the model prediction and probability

def display_prediction(pred, prob):
    st.subheader("Prediction")
    df_prob = pd.DataFrame(prob)
    df_prob.columns = ['No Kidney Disease', 'Kidney Disease']
    if pred[0] == 0:
        prob = "{:.3f}".format(prob[0][0])
        st.success(f"Kidney Disease: NO")
    else:
        prob = "{:.3f}".format(prob[0][0])
        st.error(f"Kidney Disease: YES")

    st.subheader('Prediction Probability')
    st.dataframe(df_prob,
                 column_config={
                     'No Kidney Disease': st.column_config.ProgressColumn(
                         'No Kidney Disease',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Kidney Disease': st.column_config.ProgressColumn(
                         'Kidney Disease',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                 }, hide_index=True)


# Application title and description
st.title('RenalInsight🔍')
st.write(":blue[***Monitor your kidney health with confidence 💧🩺***]")
st.write(
    "RenalInsight helps you stay on top of your kidney health by analyzing your medical data to predict the risk of "
    "kidney disease. With quick and accurate predictions powered by machine learning, it’s your personal health "
    "assistant for better kidney care 🌟")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease/data).', icon='ℹ️')

# Show Disclaimer
with st.expander('Disclaimer', icon=':material/info:'):
    display_disclaimer()

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = kidney_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Kidney', 'binary')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# Display Confusion Matrix
labels = ['Disease', 'Not Disease']
display_confusion_matrix(cm, labels)
