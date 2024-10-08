import streamlit as st
import pandas as pd
from util import *
from models import *


# Function to display the model prediction and probability

def display_prediction(pred, prob):
    st.subheader("Prediction")
    df_prob = pd.DataFrame(prob)
    df_prob.columns = ['No Fire', 'Fire']
    if pred[0] == 0:
        prob = "{:.3f}".format(prob[0][0])
        st.success(f"Fire Status: NO")
    else:
        prob = "{:.3f}".format(prob[0][0])
        st.error(f"Fire Status: YES")

    st.subheader('Prediction Probability')
    st.dataframe(df_prob,
                 column_config={
                     'No Fire': st.column_config.ProgressColumn(
                         'No Fire',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Fire': st.column_config.ProgressColumn(
                         'Fire',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                 }, hide_index=True)


# Application title and description
st.title('FireGuard')
st.write(":blue[***Stay alert to prevent forest fires 🌲🔥***]")
st.write("FireGuard analyzes environmental data to predict the likelihood of forest fires. Using advanced predictive "
         "algorithms, it offers timely alerts and insights to help you stay prepared and protect natural areas from "
         "potential wildfires 🚒🌳")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset).', icon='ℹ️')

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = algerian_forest_fire_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Algerian Forest Fire', 'binary')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# # Display Confusion Matrix
labels = ['Fire', 'Not Fire']
display_confusion_matrix(cm, labels)
