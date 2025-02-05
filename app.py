import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px

# Set Streamlit configuration
st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="🔒",  # Set the lock emoji as the page icon
    layout="wide",
    initial_sidebar_state="expanded"
)
# Static Title in Sidebar
st.title("🔒 Insider Threat Detection")  # Static title

# Sidebar for model selection
st.sidebar.header("Step 1: Choose Anomaly Detection Model")
model_option = st.sidebar.selectbox(
    "Select a Machine Learning Model for Anomaly Detection:",
    ("Isolation Forest", "Local Outlier Factor", "One-Class SVM")
)

# Sidebar for detection methods
st.sidebar.header("Step 2: Choose Detection Method")
detection_method = st.sidebar.selectbox(
    "Choose a detection method:",
    (
        "Risk Scoring",
        "Behavioral Analysis",
        "Access Monitoring",
        "File Monitoring",
        "Anomaly Detection",
    )
)

# Sidebar for dataset upload
st.sidebar.header("Step 3: Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your user behavior CSV file", type="csv")

# Main section for analysis
if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Display dataset
    st.subheader("Uploaded Dataset")
    st.dataframe(data, height=400)

    # Description of the dataset
    st.write("""
    **Dataset Overview**:  
    The uploaded dataset contains user behavior data, which includes details like login times, activity durations, and access patterns.  
    By analyzing this data, we can identify anomalies and potential insider threats based on user behavior.
    """)

    # Preprocess data: Convert timestamps and handle categorical data
    data['login_time'] = pd.to_datetime(data['login_time'])  # Convert to readable timestamp format

    # Identify categorical columns (e.g., 'access_time_of_day', 'user_role')
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Encode categorical variables using LabelEncoder
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Exclude 'login_time' from the scaling process (since it is now a datetime object)
    columns_to_scale = [col for col in data.columns if col != 'login_time']  # Exclude 'login_time'

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[columns_to_scale])  # Apply scaling only to numerical columns

    # Combine scaled data with 'login_time'
    scaled_data = pd.DataFrame(scaled_data, columns=columns_to_scale)
    scaled_data['login_time'] = data['login_time']

    # Initialize the selected model
    if model_option == "Isolation Forest":
        model = IsolationForest(contamination=0.05, random_state=42)
    elif model_option == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    elif model_option == "One-Class SVM":
        model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)

    # Train the model and predict anomalies
    model.fit(scaled_data[columns_to_scale])  # Fit only on the scaled numerical features
    predictions = model.predict(scaled_data[columns_to_scale])

    # Map predictions to anomalies
    data['Anomaly'] = predictions
    data['Anomaly'] = data['Anomaly'].apply(lambda x: -1 if x == -1 else 1)  # Normalize to -1 (anomaly) and 1 (normal)

    # Display counts of anomalies
    anomalies = data[data['Anomaly'] == -1]
    normal_users = data[data['Anomaly'] == 1]

    st.write(f"Number of Normal Users: {len(normal_users)}")
    st.write(f"Number of Anomalous Users: {len(anomalies)}")

    # Description of anomalies
    st.write("""
    **Anomaly Overview**:  
    After applying the selected anomaly detection model, the system identifies **normal** and **anomalous** users based on patterns in their behavior.  
    Anomalous users represent potential insider threats or unusual behaviors that need further investigation.
    """)

    # Visualization based on the selected detection method
    if detection_method == "Behavioral Analysis":
        st.subheader("Behavioral Analysis")
        fig = px.scatter(
            data,
            x='login_time',
            y='activity_duration',
            color='activity_duration',
            title="Activity Patterns by Login Time",
            labels={'login_time': 'Login Time', 'activity_duration': 'Activity Duration'},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("""
        **Behavioral Analysis Visualization**:  
        This scatter plot shows the relationship between **login time** and **activity duration**. Users with unusually long activity durations or odd login times might represent anomalous behavior.
        """)

    elif detection_method == "Access Monitoring":
        st.subheader("Access Monitoring")
        access_fig = px.histogram(
            data,
            x='access_time_of_day',
            color='access_time_of_day',
            title="Access Time Distribution",
            labels={'access_time_of_day': 'Access Time of Day'},
            template="plotly_dark"
        )
        st.plotly_chart(access_fig, use_container_width=True)

        st.write("""
        **Access Monitoring Visualization**:  
        This histogram illustrates the distribution of users' access times throughout the day. If a user is accessing the system at unusual hours or too frequently, it may indicate suspicious activity.
        """)

    elif detection_method == "File Monitoring":
        st.subheader("File Monitoring")
        data['file_transfer_size'] = data.get('file_transfer_size', [0] * len(data))  # Example field
        file_fig = px.box(
            data,
            y='file_transfer_size',
            points="all",
            title="File Transfer Size Distribution",
            labels={'file_transfer_size': 'File Transfer Size (MB)'},
            template="plotly_dark"
        )
        st.plotly_chart(file_fig, use_container_width=True)

        st.write("""
        **File Monitoring Visualization**:  
        This box plot shows the distribution of **file transfer sizes**. Unusual or large file transfers could be a sign of unauthorized data access or exfiltration.
        """)

    elif detection_method == "Anomaly Detection":
        st.subheader("Anomaly Detection")
        anomaly_fig = px.scatter(
            data,
            x='login_time',
            y='activity_duration',
            color='Anomaly',
            title=f"Anomalies Detected Using {model_option}",
            labels={'login_time': 'Login Time', 'activity_duration': 'Activity Duration'},
            color_discrete_map={1: 'blue', -1: 'red'},
            template="plotly_dark"
        )
        st.plotly_chart(anomaly_fig, use_container_width=True)

        st.write("""
        **Anomaly Detection Visualization**:  
        This scatter plot shows **normal** behavior (blue) and **anomalous** behavior (red) detected by the chosen machine learning model. Red points represent users exhibiting suspicious activity.
        """)

    elif detection_method == "Risk Scoring":
        st.subheader("Risk Scoring")
        data['risk_score'] = data['activity_duration'] / data['activity_duration'].max() * 100
        st.write("Risk Scores:")
        st.dataframe(data[['login_time', 'activity_duration', 'risk_score']])

        risk_fig = px.bar(
            data,
            x='login_time',
            y='risk_score',
            title="Risk Scores Over Time",
            labels={'login_time': 'Login Time', 'risk_score': 'Risk Score'},
            template="plotly_dark"
        )
        st.plotly_chart(risk_fig, use_container_width=True)

        st.write("""
        **Risk Scoring Visualization**:  
        This bar chart shows the **risk scores** assigned to users based on their activity. Higher risk scores indicate users who exhibit potentially dangerous behaviors.
        """)

    # Display anomalies
    st.subheader("Detected Anomalies")
    st.dataframe(anomalies)

else:
    st.info("Please select a model, choose a detection method, and upload a CSV file to proceed.")
