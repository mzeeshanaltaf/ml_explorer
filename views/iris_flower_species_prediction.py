import pandas as pd
import streamlit as st
from util import *
from models import *


# Function to display the model prediction and probability
def display_prediction(pred, prob):
    st.subheader("Flower Specie Prediction")

    # Encoding of Species
    species_mapper = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.success(species_mapper[pred[0]])
    df_prob = pd.DataFrame(prob)
    st.subheader('Prediction Probability')
    df_prob.columns = ['Setosa', 'Versicolor', 'Virginica']
    st.dataframe(df_prob,
                 column_config={
                     'Setosa': st.column_config.ProgressColumn(
                         'Setosa',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Versicolor': st.column_config.ProgressColumn(
                         'Versicolor',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Virginica': st.column_config.ProgressColumn(
                         'Virginica',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                 }, hide_index=True)


# Application title and description
st.title(' Iris Identifier 🌸')
st.write(":blue[***Measure the Petals, Reveal the Species!***]")
st.write("Iris Identifier uses key measurements like sepal length, sepal width, petal length, and petal width to "
         "predict the species of an Iris flower. 🌿 Simply input the details and watch as nature's beauty "
         "is classified! 🌼")

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = iris_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Iris', 'Multi Class')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# Display Confusion Matrix
labels = ['Setosa', 'Versicolr', 'Virginica']
display_multi_class_confusion_matrix(cm, labels)
