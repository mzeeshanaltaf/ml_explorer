import streamlit as st
import pandas as pd
from util import *
from models import *


# Function to display the model prediction and probability

def display_prediction(pred, prob):
    st.subheader("Prediction")
    df_prob = pd.DataFrame(prob)
    df_prob.columns = ['Not Survived', 'Survived']
    if pred[0] == 0:
        prob = "{:.3f}".format(prob[0][0])
        st.error(f"Survived: NO")
    else:
        prob = "{:.3f}".format(prob[0][0])
        st.success(f"Survived: YES")

    st.subheader('Prediction Probability')
    st.dataframe(df_prob,
                 column_config={
                     'Not Survived': st.column_config.ProgressColumn(
                         'Not Survived',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Survived': st.column_config.ProgressColumn(
                         'Survived',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                 }, hide_index=True)


# Application title and description
st.title('TitanicSurvive')
st.write(":blue[***Uncover the fate of Titanic passengers 🚢🔍***]")
st.write("TitanicSurvive uses historical passenger data to predict the likelihood of survival from the Titanic "
         "disaster. With a simple input, it provides insights into whether a passenger survived, using advanced "
         "algorithms to analyze and interpret the data 🛳️📊")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/competitions/titanic/data).', icon='ℹ️')

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = titanic_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Titanic', 'binary')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# Display Confusion Matrix
labels = ['Survived', 'Not Survived']
display_confusion_matrix(cm, labels)
