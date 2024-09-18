import streamlit as st
from util import *
from models import *


# Function to display the model prediction and probability

def display_prediction(pred, prob):
    st.subheader("Prediction")
    if pred[0] == 0:
        prob = "{:.3f}".format(prob[0][0])
        st.success(f"Heart Disease: NO")
    else:
        prob = "{:.3f}".format(prob[0][0])
        st.error(f"Heart Disease: YES")


# Application title and description
st.title('RenalInsight🔍')
st.write(":blue[***Monitor your kidney health with confidence 💧🩺***]")
st.write(
    "RenalInsight helps you stay on top of your kidney health by analyzing your medical data to predict the risk of "
    "kidney disease. With quick and accurate predictions powered by machine learning, it’s your personal health "
    "assistant for better kidney care 🌟")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease/data).', icon='ℹ️')

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = kidney_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Kidney')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# Display Confusion Matrix
display_confusion_matrix(cm)
