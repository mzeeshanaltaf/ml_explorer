# Import libraries
import streamlit as st


def display_footer():
    footer = """
    <style>
    /* Ensures the footer stays at the bottom of the sidebar */
    [data-testid="stSidebar"] > div: nth-child(3) {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
    }

    .footer {
        color: grey;
        font-size: 15px;
        text-align: center;
        background-color: transparent;
    }
    </style>
    <div class="footer">
    Made with ❤️ by <a href="mailto:zeeshan.altaf@92labs.ai">Zeeshan</a>.
    Source code <a href="https://github.com/mzeeshanaltaf/ml-multiple-disease-predictor">here</a>
    </div>
    """
    st.sidebar.markdown(footer, unsafe_allow_html=True)


# --- PAGE SETUP ---
main_page = st.Page(
    "views/main.py",
    title="MLxplorer",
    icon=":material/explore:",
    default=True,
)

heart_disease_page = st.Page(
    "views/heart_disease_predictor.py",
    title="HeartCheck ❤️",
    icon=":material/ecg_heart:",
)

diabetes_disease_page = st.Page(
    "views/diabetes_disease_predictor.py",
    title="DiabPredict🩸",
    icon=":material/glucose:",
)
kidney_disease_page = st.Page(
    "views/kidney_disease_predictor.py",
    title="RenalInsight🔍",
    icon=":material/nephrology:",
)
liver_disease_page = st.Page(
    "views/liver_disease_predictor.py",
    title="HepaScan 🧬",
    icon=":material/immunology:",
)

parkinson_disease_page = st.Page(
    "views/parkinson_disease_predictor.py",
    title="NeuroTrack️🧠",
    icon=":material/wheelchair_pickup:",
)

wine_quality_page = st.Page(
    "views/wine_quality_prediction.py",
    title="WineWise🍷",
    icon=":material/wine_bar:",
)

forest_fire_page = st.Page(
    "views/algerian_forest_fire_prediction.py",
    title="FireGuard 🌲🔥",
    icon=":material/forest:",
)

titanic_survival_page = st.Page(
    "views/titanic_survival_prediction.py",
    title="TitanicSurvive 🚢",
    icon=":material/directions_boat:",
)

penguins_species_page = st.Page(
    "views/penguins_species_prediction.py",
    title="Penguin Predictor 🐧",
    icon=":material/waves:",
)

iris_flower_species_page = st.Page(
    "views/iris_flower_species_prediction.py",
    title="Iris Identifier 🌸",
    icon=":material/local_florist:",
)

pg = st.navigation({
    "Home": [main_page],
    "Machine Learning Apps": [heart_disease_page, diabetes_disease_page, kidney_disease_page, liver_disease_page,
                              parkinson_disease_page, wine_quality_page, forest_fire_page, titanic_survival_page,
                              penguins_species_page, iris_flower_species_page]
})

# display_footer()
pg.run()
