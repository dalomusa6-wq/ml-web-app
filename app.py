import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle

lg = pickle.load(open('placement.pkl','rb'))

# web app
img = Image.open('Job-Placement-Agency.jpg')
st.image(img,width=650)
st.title("Job Placement Prediciton Model")

input_text = st.text_input("Enter all features")
if input_text:
   # 1. Split the text box inputs by comma
    raw_list = input_text.split(',')
    
    # 2. Convert 'True' to 1.0, 'False' to 0.0, and strings to numeric floats
    input_list = []
    for item in raw_list:
        clean_item = item.strip().lower()
        if clean_item == 'true':
            input_list.append(1.0)
        elif clean_item == 'false':
            input_list.append(0.0)
        else:
            input_list.append(float(clean_item))
            
    # 1. Split the text box inputs by comma
    raw_list = input_text.split(',')
    
    # 2. Convert 'True' to 1.0, 'False' to 0.0, and strings to numeric floats
    input_list = []
    for item in raw_list:
        clean_item = item.strip().lower()
        if clean_item == 'true':
            input_list.append(1.0)
        elif clean_item == 'false':
            input_list.append(0.0)
        elif clean_item != '':  
            input_list.append(float(clean_item))
            
    # 3. Safely check feature counts before predicting
    np_df = np.asarray(input_list).reshape(1, -1)
    
    expected_features = lg.n_features_in_
    actual_features = np_df.shape[1]
    
    if actual_features != expected_features:
        st.error(f"Data mismatch! Your model expects **{expected_features}** inputs, but you provided **{actual_features}** inputs. Please check your comma-separated list.")
    else:
        prediction = lg.predict(np_df)
        if prediction[0] == 1:
            st.success("🎉 This Person Is Placed")
        else:
            st.warning("⚠️ This Person is not Placed")
