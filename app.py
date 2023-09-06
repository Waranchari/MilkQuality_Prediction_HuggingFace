import joblib
import pandas as pd
import streamlit as st 

# Load your existing machine learning model
model = joblib.load('model (3).joblib')
unique_values = joblib.load('unique_values (3).joblib')

def main():
    st.title("Milk Quality Analysis")

    with st.form("questionaire"):
        pH = st.slider("pH Value", min_value=0.0, max_value=14.0, step=0.1)
        Temperature = st.slider("Temperature", min_value=0, max_value=100)
        Taste = st.selectbox("Taste", unique_values["Taste"])
        Odor = st.selectbox("Odor", unique_values["Odor"])
        Fat = st.selectbox("Fat", unique_values["Fat"])
        Turbidity = st.selectbox("Turbidity", unique_values["Turbidity"])
        Colour = st.slider("Colour", min_value=0, max_value=255)

        clicked = st.form_submit_button("Predict Quality")
        if clicked:
            # Use your existing model to make predictions
            result = model.predict(pd.DataFrame({"pH": [pH],
                                                 "Temperature": [Temperature],
                                                 "Taste": [Taste],
                                                 "Odor": [Odor],
                                                 "Fat": [Fat],
                                                 "Turbidity": [Turbidity],
                                                 "Colour": [Colour]}))
            result = result[0]
            st.success('The predicted quality is {}'.format(result))

if __name__ == '__main__':
    main()
