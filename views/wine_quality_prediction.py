import streamlit as st
from util import *
from models import *


# Function to display the model prediction and probability

def display_prediction(pred, prob):
    st.subheader("Prediction")
    if pred[0] == 0:
        prob = "{:.3f}".format(prob[0][0])
        st.error(f"Wine Quality: Bad")
    else:
        prob = "{:.3f}".format(prob[0][0])
        st.success(f"Wine Quality: Good")


# Application title and description
st.title('WineWise🍷')
st.write(":blue[***Discover the quality of your wine with a click 🥂🔍***]")
st.write("WineWise analyzes your input data to predict the quality of wine. Using advanced algorithms, it provides "
         "quick and reliable assessments, helping you choose the perfect bottle for any occasion 🍷✨")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset).', icon='ℹ️')

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = wine_quality_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Wine')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# Display Confusion Matrix
display_confusion_matrix(cm)

