import pandas as pd
import streamlit as st
from util import *
from models import *


# Function to display the model prediction and probability
def display_prediction(pred, prob):
    st.subheader("Penguin Specie Prediction")

    # Encoding of Species
    species_mapper = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    st.success(species_mapper[pred[0]])
    df_prob = pd.DataFrame(prob)
    st.subheader('Prediction Probability')
    df_prob.columns = ['Adelie', 'Chinstrap', 'Gentoo']
    st.dataframe(df_prob,
                 column_config={
                     'Adelie': st.column_config.ProgressColumn(
                         'Adelie',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Chinstrap': st.column_config.ProgressColumn(
                         'Chinstrap',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Gentoo': st.column_config.ProgressColumn(
                         'Gentoo',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                 }, hide_index=True)


# Application title and description
st.title('Penguin Predictor 🐧')
st.write(":blue[***Measure, Input, and Discover the Species!***]")
st.write("Penguin Predictor takes key measurements like bill length, bill depth, flipper length, body mass, and sex to "
         "accurately predict which species your penguin belongs to. Unlock the secrets of these fascinating birds with "
         "just a few inputs! 🌿✨")

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = penguin_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Penguin', 'Multi Class')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# Display Confusion Matrix
labels = ['Adelie', 'Gentoo', 'Chinstrap']
display_multi_class_confusion_matrix(cm, labels)
