import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
#from pycaret.classification import *
import base64



#----------------------------------------------------------------------------------------------
#----NavigationBar
#----------------------------------------------------------------------------------------------





with st.sidebar:
	logo = Image.open("logo.png")
	st.image(logo)
	
	selected = option_menu(
		menu_title="Menu",
		options=["Home","Prediction","Model Monitoring","Our Team"],
		icons=["house","graph-up-arrow","file-earmark-bar-graph","info-circle"],
		menu_icon="cast",
		default_index=0,
	)





#----------------------------------------------------------------------------------------------
#----HOME
#----------------------------------------------------------------------------------------------



if selected == "Home":
	banner = Image.open("AD-SQUAD.png")
	st.image(banner)
	def load_lottieurl(url: str):
		r = requests.get(url)
		if r.status_code != 200:
			return None
		return r.json()

	anime = "https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"
	anime_json = load_lottieurl(anime)
	st_lottie(anime_json)


# st.markdown("<h1 style='text-align: center;'>Loan Default Prediction App</h1>", unsafe_allow_html=True)

# st_lottie(lottie, height=250)

# st.write("""
# Welcome to the **Loan Default Prediction System**.  
# This app helps financial institutions instantly predict the likelihood of customer loan defaults using advanced machine learning models.
# """)




#----------------------------------------------------------------------------------------------
#----PREDICTION
#----------------------------------------------------------------------------------------------




if selected == "Prediction":
    st.title("Upload Loan Data for Prediction")

    # Upload file
    data = st.file_uploader("Upload your CSV file", type=["csv"])

    if data is not None:
        df = pd.read_csv(data)

        st.subheader("Uploaded Data")
        st.write(df.head())

        # Load the pipeline
        import joblib
        pipeline = joblib.load("loan_pipeline.pkl")

        # Predict
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)[:, 1]

        # Prepare output
        result_df = df.copy()
        result_df["default_prediction"] = predictions
        result_df["default_probability"] = probabilities

        st.subheader("Prediction Results")
        st.write(result_df)

        # Download button
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="loan_predictions.csv",
            mime="text/csv"
        )




#----------------------------------------------------------------------------------------------
#----REPORT
#----------------------------------------------------------------------------------------------



if selected == "Model Monitoring":
	st.title("Model Monitoring")
	def load_lottieurl(url: str):
		r = requests.get(url)
		if r.status_code != 200:
			return None
		return r.json()

	shop_anime = "https://assets3.lottiefiles.com/private_files/lf30_y9czxcb9.json"
	shop_anime_json = load_lottieurl(shop_anime)
	st_lottie(shop_anime_json)




#----------------------------------------------------------------------------------------------
#----CONTACT
#----------------------------------------------------------------------------------------------

if selected == "Our Team":
    import streamlit as st
    from streamlit_lottie import st_lottie
    import requests

    st.markdown("<h1 style='text-align:center; color:#f9753f;'>Meet the Team</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#f9c13f;'>The Alpha Delta Squad behind the Loan Default Prediction App</p>", unsafe_allow_html=True)

    # Load Lottie animation
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    anime_url = "https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"
    anime_json = load_lottieurl(anime_url)

    # Create two columns for team list + animation
    col_left, col_right = st.columns([2, 1])

    # Left column: Team list with color accents
    with col_left:
        st.markdown("""
        <ul style='color:#FFFFFF; font-size:16px; line-height:1.8;'>
            <li style='color:#d74492;'>1. Mustapha Abdallah - 22424206</li>
            <li style='color:#e5532d;'>2. Emmanuel Oteng Wilson - 22425111</li>
            <li style='color:#f9753f;'>3. Florence Manubea Affoh - 22428906</li>
            <li style='color:#f9c13f;'>4. Daniel Karikari - 22424563</li>
            <li style='color:#d74492;'>5. Michael Opoku - 22427541</li>
            <li style='color:#e5532d;'>6. Desmond Techie - 22424555</li>
            <li style='color:#f9753f;'>7. Delight Sefiamor Akoe - 22424698</li>
            <li style='color:#f9c13f;'>8. Saxel Awuku Yeboah - 22424842</li>
            <li style='color:#d74492;'>9. Godwin Baah - 22424736</li>
            <li style='color:#e5532d;'>10. Vanessa Atta-Fynn - 22425700</li>
        </ul>
        """, unsafe_allow_html=True)

    # Right column: Lottie animation
    with col_right:
        st_lottie(anime_json, height=300, key="team_animation")



	






























