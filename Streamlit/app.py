import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
#from pycaret.classification import *
import base64



#----------------------------------------------------------------------------------------------
#----NavigationBar
#----------------------------------------------------------------------------------------------





with st.sidebar:
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
	st.write("""
	## Loan Default Prediction
	Developed by: **Alpha Delta Squad**
	""")




#----------------------------------------------------------------------------------------------
#----PREDICTION
#----------------------------------------------------------------------------------------------




if selected == "Prediction":
	st.title("Upload Data")

	data= st.file_uploader("Choose a file")
	if data is not None:
		df = pd.read_csv(data, encoding='ISO-8859-1')
		
		#Splitting data

		train_data = df.sample(frac=0.9, random_state=123)
		test_data = df.drop(train_data.index)
		test_data.reset_index(drop=True, inplace=True)

		st.subheader("Train data")
		st.write(train_data)

		#classify = setup(data=train_data, target='Churn',
            #transformation=True,remove_outliers=True,normalize=True,feature_interaction=True, feature_selection=True,
            #remove_multicollinearity=True,fix_imbalance=True,silent=True, session_id=123)

		#best_model = compare_models()
		#tuned_model = tune_model(best_model, optimize='AUC')
		#finalize_model(tuned_model)

		st.subheader("Test Data")

		loaded_model = load_model("Churn_Model")

		pred = predict_model(loaded_model, data=test_data)
		st.subheader("Classification")
		st.write(pred)




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
	st.title("Meet the Team")
	st.markdown("View Notebook...")
	st.write("""
		  ## Alpha Delta Squad
		  
	1. Mustapha Abdallah - 22424206  
    2. Emmanuel Oteng Wilson - 22425111  
    3. Florence Manubea Affoh- 22428906
    4. Daniel karikari - 22424563
    5. Michael Opoku - 22427541
    6. Desmond Techie - 22424555
    7. Delight Sefiamor Akoe - 22424698
    8. Saxel Awuku Yeboah- 22424842
    9. Godwin Baah - 22424736
	10. Vanessa Atta-Fynn - 22425700
	""")
	






























