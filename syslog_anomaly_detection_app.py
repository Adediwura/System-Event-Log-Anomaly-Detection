import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained One-Class SVM model
model = joblib.load('ocsvm_model.pkl')

# Title of the Streamlit app
st.title("System Event Log Anomaly Detection")

# Option to choose between manual entry or batch processing
option = st.selectbox(
    "How would you like to provide the data?",
    ("Manual Entry", "Upload CSV File")
)

if option == "Manual Entry":
    st.header("Manual Feature Entry for Prediction")
    st.write("Enter the values for each feature to get a prediction.")

    # Example feature names; replace with actual feature names from your model
    feature_names = ['event_id', 'category', 'event_type']  # Ensure these match your model's training features
    input_data = []

    # Create input fields dynamically based on feature names
    for feature in feature_names:
        value = st.number_input(f"Enter value for {feature}", value=0.0)
        input_data.append(value)

    # Convert the list of inputs to a numpy array and reshape for prediction
    input_data = np.array(input_data).reshape(1, -1)

    # Predict the result using the loaded model
    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction == -1:
            st.error("This input is classified as an Anomaly!")
        else:
            st.success("This input is classified as Normal.")

elif option == "Upload CSV File":
    st.header("Batch Anomaly Detection from CSV")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_log = pd.read_csv(uploaded_file)
        
        # Ensure the CSV has the expected features
        expected_features = ['event_id', 'category', 'event_type']  # Match these with your model
        if not all(feature in df_log.columns for feature in expected_features):
            st.error(f"Uploaded CSV file must contain the following columns: {expected_features}")
        else:
            X = df_log[expected_features].values

            # Predict anomalies using the loaded model
            predictions = model.predict(X)
            df_log['anomaly'] = predictions

            # Display the results
            st.subheader("Anomaly Detection Results")
            st.write(df_log)

            # Filter and display the detected anomalies
            anomalies = df_log[df_log['anomaly'] == -1]
            st.subheader(f"Number of Anomalies Detected: {len(anomalies)}")
            st.write(anomalies)

            # Option to download the results with anomalies marked
            csv = df_log.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='anomaly_detection_results.csv',
                mime='text/csv',
            )
    else:
        st.info("Please upload a CSV file to analyze for anomalies.")
