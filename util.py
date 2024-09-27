import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# List of supported Machine Learning Models
supported_models = ["Logistic Regression", "Support Vector Machines", 'K-Nearest Neighbor', "Decision Tree",
                    "Random Forest", 'AdaBoost', 'Gradient Boost', 'XGBoost', "Gaussian NB"]


def heart_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    gender_dic = {'Male': 1, 'Female': 0}
    chest_paint_dict = {'Asymptomatic': 0, 'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3}
    fasting_sugar_dict = {'True': 1, 'False': 0}
    resting_ecg_dict = {'Normal': 0, 'ST-TWave Abnormality': 1, 'LVH': 2}
    angina_dict = {'Yes': 1, 'No': 0}
    slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thallium_dict = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    no_of_vessels_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

    # Get input parameters from user
    input_dict = {}
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        age = col1.slider('Age (Years)', min_value=20, max_value=90, value=40)
        sex = col2.selectbox('Gender', ('Male', 'Female'))
        cp = col3.selectbox('Chest Pain Type',
                            ('Asymptomatic', 'Typical Angina', 'Atypical Angina', 'Non-anginal Pain'))

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        rbp = col1.slider('Resting BP (mm Hg)', min_value=40, max_value=200, value=120)
        cholesterol = col2.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        fbs = col3.selectbox('Fasting Blood Sugar > 120 (mg/dl)', ('True', 'False'))

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        resting_ecg = col1.selectbox('Resting ECG', ('Normal', 'ST-TWave Abnormality', 'LVH'))
        max_hr = col2.number_input('Max Heart Rate', min_value=40, max_value=205, value=150)
        ang = col3.selectbox('Angina', ('Yes', 'No'))

    with st.container(border=True):
        col1, col2 = st.columns(2)
        old_peak = col1.number_input('ST Depression induced by exercise', min_value=0.0, max_value=7.0, value=2.5)
        slope = col2.selectbox('Slope', ('Upsloping', 'Flat', 'Downsloping'))
    with st.container(border=True):
        col1, col2 = st.columns(2)
        ca = col1.selectbox('No. of Major Vessels colored by Fluoroscopy', ('0', '1', '2', '3', '4'))
        thallium = col2.selectbox('Thallium Stress Test', ('Normal', 'Fixed Defect', 'Reversible Defect'))

    # Update the input dictionary with user selected values
    input_dict['Age'] = age
    input_dict['Sex'] = gender_dic[sex]
    input_dict['CP'] = chest_paint_dict[cp]
    input_dict['rbp'] = rbp
    input_dict['chol'] = cholesterol
    input_dict['fbs'] = fasting_sugar_dict[fbs]
    input_dict['recg'] = resting_ecg_dict[resting_ecg]
    input_dict['mhr'] = max_hr
    input_dict['angina'] = angina_dict[ang]
    input_dict['old_peak'] = old_peak
    input_dict['slope'] = slope_dict[slope]
    input_dict['ca'] = no_of_vessels_dict[ca]
    input_dict['thal'] = thallium_dict[thallium]

    return input_dict


# Get diabetes data from the user
def diabetes_input_parameters():
    # Get input parameters from user
    input_dict = {}

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        age = col1.slider('Age (Years)', min_value=20, max_value=90, value=40)
        pregnancies = col2.slider('Pregnancies', min_value=0, max_value=20, value=2)
        glucose = col3.slider('Glucose (mg/dL)', min_value=40, max_value=200, value=80)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        bp = col1.slider('Blood Pressure (mm Hg)', min_value=20, max_value=122, value=80)
        skin_thickness = col2.slider('Skin Thickness (mm)', min_value=5, max_value=100, value=30)
        insulin = col3.slider('Insulin (mu U/ml)', min_value=14, max_value=1000, value=150)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        bmi = col1.slider('BMI', min_value=15, max_value=70, value=30)
        dpf = col2.slider('Diabetes Pedigree Function', min_value=0.05, max_value=2.50, value=0.5)

    # Update the input dictionary with user selected values
    input_dict['pregnancies'] = pregnancies
    input_dict['glucose'] = glucose
    input_dict['blood_pressure'] = bp
    input_dict['skin_thickness'] = skin_thickness
    input_dict['insulin'] = insulin
    input_dict['bmi'] = bmi
    input_dict['dpf'] = dpf
    input_dict['age'] = age

    return input_dict


# Get liver data from the user
def liver_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    gender_dic = {'Male': 1, 'Female': 0}

    # Get input parameters from user
    input_dict = {}
    with st.container(border=True):
        col1, col2 = st.columns(2)
        age = col1.slider('Age (Years)', min_value=4, max_value=90, value=40)
        gender = col2.selectbox('Gender', ('Male', 'Female'))

    with st.container(border=True):
        col1, col2 = st.columns(2)
        total_bilirubin = col1.number_input('Total Bilirubin (mg/dL)', min_value=0.1, max_value=25.0, value=7.3)
        direct_bilirubin = col2.number_input('Direct Bilirubin (mg/dL)', min_value=0.0, max_value=10.0, value=2.1)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        alkaline_phosphatase = col1.number_input('Alkaline Phosphatase (IU/L)', min_value=40, max_value=900, value=100)
        alanine_aminotransferase = col2.number_input('Alanine Aminotransferase (IU/L)', min_value=5, max_value=1000,
                                                     value=35)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        aspartate_aminotransferase = col1.number_input('Aspartate Aminotransferase (IU/L)', min_value=10,
                                                       max_value=1000,
                                                       value=50)
        total_proteins = col2.number_input('Total Proteins (g/dL)', min_value=2.7, max_value=9.6, value=7.0)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        albumin = col1.number_input('Albumin (g/dL)', min_value=0.9, max_value=5.5, value=3.2)
        albumin_globulin_ratio = col2.number_input('Albumin Globulin Ratio', min_value=0.3, max_value=2.8, value=1.0)

    # Update the input dictionary with user selected values
    input_dict['age'] = age
    input_dict['gender'] = gender_dic[gender]
    input_dict['total_bilirubin'] = total_bilirubin
    input_dict['direct_bilirubin'] = direct_bilirubin
    input_dict['alkaline_phosphotase'] = alkaline_phosphatase
    input_dict['alamine_aminotransferase'] = alanine_aminotransferase
    input_dict['aspartate_aminotransferase'] = aspartate_aminotransferase
    input_dict['total_proteins'] = total_proteins
    input_dict['albumin'] = albumin
    input_dict['albumin_globulin_ratio'] = albumin_globulin_ratio

    return input_dict


# Get kidney data from the user
def kidney_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    rbc_dict = {'Abnormal': 0, 'Normal': 0}
    pus_cells_dict = {'Abnormal': 0, 'Normal': 0}
    pus_cells_clumps_dict = {'Not Present': 0, 'Present': 1}
    bacteria_dict = {'Not Present': 0, 'Present': 1}
    hypertension_dict = {'No': 0, 'Yes': 1}
    diabetes_mellitus_dict = {'No': 0, 'Yes': 1}
    coronary_artery_disease_dict = {'No': 0, 'Yes': 1}
    appetite_dict = {'Good': 0, 'Poor': 1}
    peda_edema_dict = {'No': 0, 'Yes': 1}
    anemia_dict = {'No': 0, 'Yes': 1}

    # Get input parameters from user
    input_dict = {}

    with st.container(border=True):
        col1, col2 = st.columns(2)
        age = col1.slider('Age (Years)', min_value=2, max_value=90, value=40)
        blood_pressure = col2.number_input('Blood Pressure (mm/Hg)', min_value=50.0, max_value=180.0, value=76.0)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        specific_gravity = col1.number_input('Specific Gravity', min_value=1.005, max_value=1.025, value=1.015)
        albumin = col2.number_input('Albumin', min_value=0.0, max_value=5.0, value=1.0)
        sugar = col3.number_input('Sugar', min_value=0.0, max_value=5.0, value=0.5)

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        red_blood_cells = col1.selectbox('Red Blood Cells', ('Normal', 'Abnormal'))
        pus_cells = col2.selectbox('Pus Cells', ('Normal', 'Abnormal'))
        pus_cells_clumps = col3.selectbox('Pus Cells Clumps', ('Not Present', 'Present'))
        bacteria = col4.selectbox('Bacteria', ('Not Present', 'Present'))

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        blood_glucose_random = col1.number_input('Blood Glucose Random (mgs/dl)', min_value=22.0, max_value=500.0,
                                                 value=150.0)
        blood_urea = col2.number_input('Blood Urea (mgs/dl)', min_value=1.5, max_value=400.0, value=57.5)
        serum_creatinine = col3.number_input('Serum Creatinine (mgs/dl)', min_value=0.4, max_value=76.0, value=3.0)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        sodium = col1.number_input('Sodium (mEq/L)', min_value=4.5, max_value=165.0, value=137.0)
        potassium = col2.number_input('Potassium (mEq/L)', min_value=2.5, max_value=47.0, value=4.5)
        haemoglobin = col3.number_input('Haemoglobin (g)', min_value=3.0, max_value=18.0, value=12.5)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        packed_cell_volume = col1.number_input('Packed Cell Volume', min_value=9.0, max_value=54.0, value=35.0)
        white_blood_cell_count = col2.number_input('White Blood Cell Count (cells/cumm)', min_value=2200.0,
                                                   max_value=27000.0, value=9000.0)
        red_blood_cell_count = col3.number_input('Red Blood Cell Count (millions/cmm)', min_value=2.0, max_value=8.0,
                                                 value=4.5)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        hypertension = col1.selectbox('hypertension', ('No', 'Yes'))
        diabetes_mellitus = col2.selectbox('Diabetes Mellitus', ('No', 'Yes'))
        coronary_artery_disease = col3.selectbox('Coronary Artery Disease', ('No', 'Yes'))

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        appetite = col1.selectbox('Appetite', ('Good', 'Poor'))
        peda_edema = col2.selectbox('Peda Edema', ('No', 'Yes'))
        anemia = col3.selectbox('Anemia', ('No', 'Yes'))

    # Update the input dictionary with user selected values
    input_dict['age'] = age
    input_dict['blood_pressure'] = blood_pressure
    input_dict['specific_gravity'] = specific_gravity
    input_dict['albumin'] = albumin
    input_dict['sugar'] = sugar
    input_dict['red_blood_cells'] = rbc_dict[red_blood_cells]
    input_dict['pus_cells'] = pus_cells_dict[pus_cells]
    input_dict['pus_cells_clumps'] = pus_cells_clumps_dict[pus_cells_clumps]
    input_dict['bacteria'] = bacteria_dict[bacteria]
    input_dict['blood_glucose_random'] = blood_glucose_random
    input_dict['blood_urea'] = blood_urea
    input_dict['serum_creatinine'] = serum_creatinine
    input_dict['sodium'] = sodium
    input_dict['potassium'] = potassium
    input_dict['haemoglobin'] = haemoglobin
    input_dict['packed_cell_volume'] = packed_cell_volume
    input_dict['white_blood_cell_count'] = white_blood_cell_count
    input_dict['red_blood_cell_count'] = red_blood_cell_count
    input_dict['hypertension'] = hypertension_dict[hypertension]
    input_dict['diabetes_mellitus'] = diabetes_mellitus_dict[diabetes_mellitus]
    input_dict['coronary_artery_disease'] = coronary_artery_disease_dict[coronary_artery_disease]
    input_dict['appetite'] = appetite_dict[appetite]
    input_dict['peda_edema'] = peda_edema_dict[peda_edema]
    input_dict['anemia'] = anemia_dict[anemia]

    return input_dict


# Get parkinson data from the user
def parkinson_input_parameters():
    # Get input parameters from user
    input_dict = {}

    with st.container(border=True):
        st.write('Vocal Fundamental Frequency:')
        col1, col2, col3 = st.columns(3)
        freq_avg = col1.number_input('Average', min_value=80, max_value=280, value=150)
        freq_hi = col2.number_input('Maximum', min_value=100, max_value=600, value=197)
        freq_lo = col3.number_input('Minimum', min_value=60, max_value=250, value=115)

    with st.container(border=True):
        st.write('Variation in Fundamental Frequency:')
        col1, col2, col3, col4, col5 = st.columns(5)
        jitter_percent = col1.number_input('MDVP:Jitter(%)', min_value=0.001, max_value=0.008, value=0.003)
        jitter_abs = col2.number_input('MDVP:Jitter(Abs)', min_value=0.000007, max_value=0.0003, value=0.00004)
        mdvp_rap = col3.number_input('MDVP:RAP', min_value=0.0006, max_value=0.03, value=0.003)
        mdvp_ppq = col4.number_input('MDVP:PPQ', min_value=0.0008, max_value=0.02, value=0.003)
        jitter_ddp = col5.number_input('Jitter:DDP', min_value=0.002, max_value=0.07, value=0.009)

    with st.container(border=True):
        st.write('Variation in Amplitude:')
        col1, col2, col3 = st.columns(3)
        shimmer = col1.number_input('MDVP:Shimmer', min_value=0.009, max_value=0.12, value=0.03)
        shimmer_db = col2.number_input('MDVP:Shimmer(dB)', min_value=0.08, max_value=1.5, value=0.25)
        shimmer_apq3 = col3.number_input('Shimmer:APQ3', min_value=0.004, max_value=0.06, value=0.01)

        col1, col2, col3 = st.columns(3)
        shimmer_apq5 = col1.number_input('Shimmer:APQ5', min_value=0.005, max_value=0.08, value=0.015)
        mdvp_apq = col2.number_input('MDVP:APQ', min_value=0.007, max_value=0.15, value=0.025)
        shimmer_dda = col3.number_input('Shimmer:DDA', min_value=0.01, max_value=0.17, value=0.05)

    with st.container(border=True):
        st.write('Ratio of noise to tonal components in the voice:')
        col1, col2 = st.columns(2)
        nhr = col1.number_input('NHR', min_value=0.0006, max_value=0.35, value=0.025)
        hnr = col2.number_input('HNR', min_value=8.0, max_value=35.0, value=22.0)

    with st.container(border=True):
        st.write('Dynamic Complexity Measures:')
        col1, col2 = st.columns(2)
        rpde = col1.number_input('RPDE', min_value=0.2, max_value=0.7, value=0.5)
        d2 = col2.number_input('D2', min_value=1.2, max_value=4.0, value=2.5)

    with st.container(border=True):
        st.write('Signal Fractal Scaling Exponent / Fundamental Freq Variation:')
        col1, col2 = st.columns(2)
        dfa = col1.number_input('DFA', min_value=0.5, max_value=0.9, value=0.72)
        spread1 = col2.number_input('Spread1', min_value=-8.0, max_value=-2.0, value=-5.5)
        spread2 = col1.number_input('Spread2', min_value=0.006, max_value=0.5, value=0.22)
        ppe = col2.number_input('PPE', min_value=0.04, max_value=0.6, value=0.2)

        # Update the input dictionary with user selected values
        input_dict['freq_avg'] = freq_avg
        input_dict['freq_hi'] = freq_hi
        input_dict['freq_lo'] = freq_lo
        input_dict['jitter_percent'] = jitter_percent
        input_dict['jitter_abs'] = jitter_abs
        input_dict['mdvp_rap'] = mdvp_rap
        input_dict['mdvp_ppq'] = mdvp_ppq
        input_dict['jitter_ddp'] = jitter_ddp
        input_dict['shimmer'] = shimmer
        input_dict['shimmer_db'] = shimmer_db
        input_dict['shimmer_apq3'] = shimmer_apq3
        input_dict['shimmer_apq5'] = shimmer_apq5
        input_dict['mdvp_apq'] = mdvp_apq
        input_dict['shimmer_dda'] = shimmer_dda
        input_dict['nhr'] = nhr
        input_dict['hnr'] = hnr
        input_dict['rpde'] = rpde
        input_dict['dfa'] = dfa
        input_dict['spread1'] = spread1
        input_dict['spread2'] = spread2
        input_dict['d2'] = d2
        input_dict['ppe'] = ppe

        return input_dict


# Get wine quality data from the user
def wine_quality_input_parameters():
    # Get input parameters from user
    input_dict = {}

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        fixed_acidity = col1.number_input('Fixed Acidity', min_value=4.0, max_value=16.0, value=7.5, step=1.0)
        volatile_acidity = col2.number_input('Volatile Acidity', min_value=0.1, max_value=1.6, value=0.5, step=0.1)
        citric_acid = col3.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        residual_sugar = col1.number_input('Residual Sugar', min_value=0.5, max_value=16.0, value=2.5, step=1.0)
        chlorides = col2.number_input('Chlorides', min_value=0.01, max_value=0.65, value=0.08, step=0.05)
        free_sulfur_dioxide = col3.number_input('Free Sulfur Dioxide', min_value=1.0, max_value=72.0, value=16.0,
                                                step=1.0)
        total_sulfur_dioxide = col4.number_input('Total Sulfur Dioxide', min_value=6.0, max_value=290.0, value=45.0,
                                                 step=1.0)

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        density = col1.number_input('Density', min_value=0.990, max_value=1.05, value=0.996)
        ph = col2.number_input('pH', min_value=2.0, max_value=4.0, value=3.5, step=0.1)
        sulphate = col3.number_input('Sulphate', min_value=0.3, max_value=2.0, value=0.65, step=0.1)
        alcohol = col4.number_input('Alcohol', min_value=8.0, max_value=15.0, value=10.5, step=1.0)

    # Update the input dictionary with user selected values
    input_dict['fixed_acidity'] = fixed_acidity
    input_dict['volatile_acidity'] = volatile_acidity
    input_dict['citric_acid'] = citric_acid
    input_dict['residual_sugar'] = residual_sugar
    input_dict['chlorides'] = chlorides
    input_dict['free_sulfur_dioxide'] = free_sulfur_dioxide
    input_dict['total_sulfur_dioxide'] = total_sulfur_dioxide
    input_dict['density'] = density
    input_dict['ph'] = ph
    input_dict['sulphate'] = sulphate
    input_dict['alcohol'] = alcohol

    return input_dict


# Get titanic data from the user
def titanic_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    gender_dic = {'Male': 1, 'Female': 0}
    ticket_dic = {'1': 1, '2': 2, '3': 3}
    port = {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}

    # Get  input parameters from user
    input_dict = {}
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        age = col1.slider('Age (Years)', min_value=1, max_value=90, value=40)
        gender = col2.selectbox('Gender', ('Male', 'Female'))
        ticket_class = col3.selectbox('Ticket Class', ('1', '2', '3'))

    with st.container(border=True):
        col1, col2 = st.columns(2)
        no_of_siblings = col1.slider('Number of Siblings', min_value=0, max_value=8, value=2)
        no_of_parents_children = col2.slider('Number of Parents Children', min_value=0, max_value=6, value=2)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        fare = col1.number_input('Passenger Fare', min_value=0.0, max_value=515.0, value=30.0, step=1.0)
        embarkation_port = col2.selectbox('Port of Embarkation', ('Cherbourg', 'Queenstown', 'Southampton'))

    # Update the input dictionary with user selected values
    input_dict['ticket_class'] = ticket_dic[ticket_class]
    input_dict['age'] = age
    input_dict['no_of_siblings'] = no_of_siblings
    input_dict['no_of_parents_children'] = no_of_parents_children
    input_dict['fare'] = fare
    input_dict['gender'] = gender_dic[gender]
    if port[embarkation_port] == 'Q':
        input_dict['embarkation_port_q'] = True
        input_dict['embarkation_port_s'] = False
    elif port[embarkation_port] == 'S':
        input_dict['embarkation_port_q'] = False
        input_dict['embarkation_port_s'] = True
    else:
        input_dict['embarkation_port_q'] = False
        input_dict['embarkation_port_s'] = False

    # df = pd.DataFrame([input_dict])

    return input_dict


# Get Algerian Forest Fire data from the user
def algerian_forest_fire_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    region_dic = {'Sidi-Bel': 1, 'Bejaia': 0}

    # Get input parameters from user
    input_dict = {}

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        region = col1.selectbox('Region', ['Sidi-Bel', 'Bejaia'])
        temperature = col2.slider('Temperature (°C)', min_value=10, value=25, max_value=45)
        rh = col3.slider('Relative Humidity (%)', min_value=21, value=90, max_value=100)
        ws = col4.slider('Wind Speed (KM/H)', min_value=5, value=13, max_value=30)

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        rain = col1.number_input('Rain (mm)', min_value=0.0, value=2.5, max_value=20.0, step=1.0)
        ffmc = col2.number_input('Fine Fuel Moisture Code', min_value=-28.0, value=70.0, max_value=99.0, step=1.0)
        dmc = col3.number_input('Duff Moisture Code', min_value=0.0, value=1.5, max_value=80.0, step=1.0)
        dc = col4.number_input('Drought Code', min_value=5.0, value=7.0, max_value=250.0, step=1.0)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        isi = col1.number_input('Initial Spread Index', min_value=0.0, value=2.0, max_value=30.0, step=1.0)
        bui = col2.number_input('Build Up Index', min_value=1.0, value=1.5, max_value=70.0, step=1.0)
        fwi = col3.number_input('Fire Weather Index', min_value=0.0, value=7.0, max_value=32.0, step=1.0)

    # Update the input dictionary with user selected values
    input_dict['temperature'] = temperature
    input_dict['rh'] = rh
    input_dict['ws'] = ws
    input_dict['rain'] = rain
    input_dict['ffmc'] = ffmc
    input_dict['dmc'] = dmc
    input_dict['dc'] = dc
    input_dict['isi'] = isi
    input_dict['bui'] = bui
    input_dict['fwi'] = fwi
    input_dict['region'] = region_dic[region]

    return input_dict


# Get penguin data from the user
def penguin_input_parameters():
    # Dictionaries to convert labels to their corresponding integer values
    gender_dic = {'Male': 1, 'Female': 0}

    # Get  input parameters from user
    input_dict = {}
    with st.container(border=True):
        col1, col2 = st.columns(2)
        gender = col1.selectbox('Gender', ('Male', 'Female'))
        island = col2.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))

    with st.container(border=True):
        col1, col2 = st.columns(2)
        bill_length = col1.slider('Bill Length (mm)', min_value=32.1, max_value=59.6, value=43.9)
        bill_depth = col2.slider('Bill Depth (mm)', min_value=13.1, max_value=21.5, value=17.2)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        flipper_length = col1.slider('Flipper Length (mm)', min_value=172.0, max_value=231.0, value=201.0)
        body_mass = col2.slider('Body Mass (g)', min_value=2700.0, max_value=6300.0, value=4207.0)

    # Update the input dictionary with user selected values
    input_dict['bill_length'] = bill_length
    input_dict['bill_depth'] = bill_depth
    input_dict['flipper_length'] = flipper_length
    input_dict['body_mass'] = body_mass
    input_dict['sex'] = gender_dic[gender]

    if island == 'Biscoe':
        input_dict['island_dream'] = False
        input_dict['island_torgersen'] = False
    elif island == 'Dream':
        input_dict['island_dream'] = True
        input_dict['island_torgersen'] = False
    else:
        input_dict['island_dream'] = False
        input_dict['island_torgersen'] = True

    return input_dict


# Get penguin data from the user
def iris_input_parameters():
    # Get  input parameters from user
    input_dict = {}

    with st.container(border=True):
        col1, col2 = st.columns(2)
        sepal_length = col1.slider('Sepal Length (cm)', min_value=4.3, max_value=7.9, value=5.84)
        sepal_width = col2.slider('Sepal Width (cm)', min_value=2.0, max_value=4.4, value=3.05)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        petal_length = col1.slider('Petal Length (cm)', min_value=1.0, max_value=6.9, value=3.76)
        petal_width = col2.slider('Petal Width (cm)', min_value=0.1, max_value=2.5, value=1.20)

    # Update the input dictionary with user selected values
    input_dict['sepal_length'] = sepal_length
    input_dict['sepal_width'] = sepal_width
    input_dict['petal_length'] = petal_length
    input_dict['petal_width'] = petal_width

    return input_dict


# Function to display the footer
def display_performance_metrics(df_performance_metric):
    st.subheader('Performance Metrics')
    accuracy_col, f1_score_col, precision_col, recall_col, roc_auc_score_col = st.columns(5)
    with accuracy_col:
        with st.container(border=True):
            st.metric('*Accuracy*', value=df_performance_metric['Accuracy'].iloc[0])
    with f1_score_col:
        with st.container(border=True):
            st.metric('*F1 Score*', value=float(df_performance_metric['F1 Score'].iloc[0]))
    with precision_col:
        with st.container(border=True):
            st.metric('*Precision*', value=float(df_performance_metric['Precision'].iloc[0]))
    with recall_col:
        with st.container(border=True):
            st.metric('*Recall*', value=float(df_performance_metric['Recall'].iloc[0]))
    with roc_auc_score_col:
        with st.container(border=True):
            st.metric('*ROC AUC Score*', value=float(df_performance_metric['ROC AUC Score'].iloc[0]))


# def display_confusion_matrix(cm):
#     st.subheader('Prediction Outcome Table')
#     predicted_positive = [cm[0][0], cm[1][0]]
#     predicted_negative = [cm[0][1], cm[1][1]]
#
#     # Data for confusion matrix
#     data = {'Predicted Positive': predicted_positive,  # [True Positive, False Positive]
#             'Predicted Negative': predicted_negative}  # [False Negative, True Negative]
#
#     # Creating DataFrame
#     df_confusion_matrix = pd.DataFrame(data, index=['Actual Positive', 'Actual Negative'])
#     st.dataframe(df_confusion_matrix)

def display_confusion_matrix(cm, labels):
    st.subheader('Prediction Outcome Table')

    # Get the dimensions of the heatmap
    n = cm.shape[0]

    # Data for the heatmap
    data = cm

    # Create a mask to isolate diagonal elements
    mask = np.eye(data.shape[0], dtype=bool)  # True on diagonal, False elsewhere

    # Extract non-diagonal elements
    non_diagonal_data = data[~mask]

    # Find min and max values in the non-diagonal elements
    non_diag_min = non_diagonal_data.min()
    non_diag_max = non_diagonal_data.max()

    # Create a custom colormap for red shades: start with light red, end with dark red
    light_red_to_dark_red = LinearSegmentedColormap.from_list('custom_red', ['#fff0f0', '#ff0000'])

    # Green for diagonal
    green_cmap = ListedColormap(['green'])

    # Plot the heatmap with custom coloring
    plt.figure(figsize=(6, 6))  # Adjust figure size for an nxn heatmap

    # Plot non-diagonal elements in red with correct shades based on non-diagonal values
    ax = sns.heatmap(data, mask=mask, cmap=light_red_to_dark_red, annot=True, cbar=False, square=True,
                     vmin=non_diag_min, vmax=non_diag_max, fmt='d')

    # Plot diagonal elements in green
    sns.heatmap(data, mask=~mask, cmap=green_cmap, annot=True, cbar=False, square=True, ax=ax, fmt='d')

    # Set tick labels for the x-axis and y-axis
    ax.set_xticks(np.arange(n) + 0.5)  # Position tick marks in the center of the cells
    ax.set_yticks(np.arange(n) + 0.5)

    # Set labels for X and Y tick labels dynamically
    x_labels = labels
    y_labels = labels

    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)

    # Move x-axis labels to the top
    ax.xaxis.tick_top()  # Place x-axis labels on top
    ax.tick_params(top=True, bottom=False)  # Disable bottom ticks

    # Set X-axis and Y-axis labels
    ax.set_xlabel("Predicted Values", fontsize=10)
    ax.set_ylabel("Actual Values", fontsize=10)

    # Display the heatmap
    st.pyplot(plt)


def display_multi_class_confusion_matrix(cm, labels):
    st.subheader('Prediction Outcome Table')

    # Get the dimensions of the heatmap
    n = cm.shape[0]

    # Data for the heatmap
    data = cm

    # Create a mask to isolate diagonal elements
    mask = np.eye(data.shape[0], dtype=bool)  # True on diagonal, False elsewhere

    # Extract non-diagonal elements
    non_diagonal_data = data[~mask]

    # Find min and max values in the non-diagonal elements
    non_diag_min = non_diagonal_data.min()
    non_diag_max = non_diagonal_data.max()

    # Create a custom colormap for red shades: start with light red, end with dark red
    light_red_to_dark_red = LinearSegmentedColormap.from_list('custom_red', ['#fff0f0', '#ff0000'])

    # Green for diagonal
    green_cmap = ListedColormap(['green'])

    # Plot the heatmap with custom coloring
    plt.figure(figsize=(6, 6))  # Adjust figure size for an nxn heatmap

    # Plot non-diagonal elements in red with correct shades based on non-diagonal values
    ax = sns.heatmap(data, mask=mask, cmap=light_red_to_dark_red, annot=True, cbar=False, square=True,
                     vmin=non_diag_min, vmax=non_diag_max)

    # Plot diagonal elements in green
    sns.heatmap(data, mask=~mask, cmap=green_cmap, annot=True, cbar=False, square=True, ax=ax)

    # Set tick labels for the x-axis and y-axis
    ax.set_xticks(np.arange(n) + 0.5)  # Position tick marks in the center of the cells
    ax.set_yticks(np.arange(n) + 0.5)

    # Set labels for X and Y tick labels dynamically
    x_labels = labels
    y_labels = labels

    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)

    # Move x-axis labels to the top
    ax.xaxis.tick_top()  # Place x-axis labels on top
    ax.tick_params(top=True, bottom=False)  # Disable bottom ticks

    # Set X-axis and Y-axis labels
    ax.set_xlabel("Predicted Values", fontsize=10)
    ax.set_ylabel("Actual Values", fontsize=10)

    # Display the heatmap
    st.pyplot(plt)


def display_disclaimer():
    st.warning('*This application is for information purpose only and should not be considered as medical '
               'advice or a conclusive diagnosis. Always consult a qualified healthcare professional for an '
               'accurate diagnosis and personalized medical advice.*')
