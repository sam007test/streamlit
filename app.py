import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from pandasai import Agent

# Set the page configuration
st.set_page_config(
    page_title="Sales Data Analysis",
    page_icon="favicon.ico"  # Use the path to your favicon file
)
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2.75rem;
        font-weight: bold;
        margin: 0.5em 0;
    }
    </style>
    <div class="title">Sales Data Analysis for IOCL</div>
    """,
    unsafe_allow_html=True
)

# Set your pandasai API key
os.environ["PANDASAI_API_KEY"] = "$2a$10$JU5G0VlFOTIeAa/Gr.4W5OyiLY4A/vzpuDIl3Ahkg14jLimhzTHtq"

# Function to process each Excel file
def process_excel_file(file):
    df = pd.read_excel(file, header=None)

    # Find the headers in the dataframe
    ga_header = df[df.isin(['GA']).any(axis=1)].stack().loc[lambda x: x == 'GA'].index[0]
    ro_name_header = df[df.isin(['RO Name']).any(axis=1)].stack().loc[lambda x: x == 'RO Name'].index[0]
    date_headers = df[df.isin(['Daily Sales']).any(axis=1)].stack().loc[lambda x: x == 'Daily Sales'].index

    # Extract data for GA, RO Name, and Daily Sales columns
    data_start_row = max(ga_header[0], ro_name_header[0], max(header[0] for header in date_headers)) + 1
    ga_col = ga_header[1]
    ro_name_col = ro_name_header[1]
    date_cols = [header[1] for header in date_headers]

    data = df.iloc[data_start_row:, [ga_col, ro_name_col] + date_cols].copy()

    # Define column names for the new dataframe
    columns = ['GA', 'RO Name'] + [str(df.iat[header[0] - 1, header[1]]) for header in date_headers]
    data.columns = columns

    # Drop columns with consecutive NaNs
    def drop_cols_with_consecutive_nans(df, threshold):
        return df.loc[:, df.iloc[:threshold].isna().sum() < threshold]

    data = drop_cols_with_consecutive_nans(data, 10)

    # Further cleaning of the dataframe
    data = data[~data['RO Name'].isin(['Sub-Total-1', 'Sub-Total-2'])]
    data = data[data['GA'].notna()]
    data = data.fillna(0)
    data.columns = data.columns.astype(str)
    data.replace('-', 0, inplace=True)
    data = data[data['RO Name'].ne(0)]

    return data

# XGBoost prediction function
def xgboost_prediction(data):
    # Prepare data for modeling
    data = data.melt(id_vars=['GA', 'RO Name'], var_name='Date', value_name='Sales')
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    # Create features and labels
    data['Day_of_Week'] = data['Date'].dt.dayofweek
    features = data[['RO Name', 'Day_of_Week']]
    labels = data['Sales']

    # Convert categorical variables to numeric
    features = pd.get_dummies(features, columns=['RO Name'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Calculate RMSE for training and testing
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    st.write(f"Training Root Mean Squared Error: {train_rmse:.2f}")
    st.write(f"Testing Root Mean Squared Error: {test_rmse:.2f}")

    # User input for prediction
    st.write("### Predict Sales for a Specific RO and Date")
    ro_options = data['RO Name'].unique().tolist()
    selected_ro = st.selectbox("Select RO for prediction", ro_options)
    input_date = st.date_input("Select a date for prediction")
    day_of_week = pd.Timestamp(input_date).dayofweek

    # Create a dataframe for the input date prediction
    prediction_features = pd.get_dummies(pd.DataFrame({'RO Name': [selected_ro], 'Day_of_Week': [day_of_week]}), columns=['RO Name'])
    prediction_features = prediction_features.reindex(columns=X_train.columns, fill_value=0)

    # Make prediction for the input date and RO
    prediction = model.predict(prediction_features)
    st.write(f"Predicted Sales for {selected_ro} on {input_date}: {prediction[0]:.2f}")

# Chat function using pandasai
def pandasai_chat(data):
    agent = Agent(data)
    st.write("Hello! I am your AI assistant. Please ask me any questions about the sales data. For example, you can ask 'Which are the top 5 countries by sales?'")
    instructions = "Please note that the answers are based on the cleaned sales data provided."
    st.write(instructions)

    user_input = st.text_input("Ask a question about the sales data:")
    if user_input:
        response = agent.chat(user_input)

        if isinstance(response, str) and response.endswith('.png'):
            # Display the image
            st.image(response)
            # Delete the file after rendering
            os.remove(response)
        else:
            st.write(response)

# Streamlit application
# st.title('Sales Data Analysis')

uploaded_files = st.file_uploader("Upload Excel files", type="xlsx", accept_multiple_files=True)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        data = process_excel_file(file)
        all_data.append(data)

    combined_df = pd.concat(all_data).groupby(['GA', 'RO Name']).sum().reset_index()

    # Add a menu to choose between RO-wise, GA-wise analysis, prediction, and chat with AI
    analysis_type = st.radio("Choose analysis type", ('RO-wise', 'GA-wise', 'XGBoost Prediction', 'Chat with AI'))

    if analysis_type == 'RO-wise':
        # RO-wise analysis
        ro_options = combined_df['RO Name'].unique().tolist()
        selected_ros = st.multiselect("Select ROs for analysis", ro_options)

        if not selected_ros:
            st.write("Please select at least one RO for analysis.")
        else:
            ro_data = combined_df[combined_df['RO Name'].isin(selected_ros)].copy()

            # Melt the dataframe
            ro_data_melted = ro_data.melt(id_vars=['GA', 'RO Name'], var_name='Date', value_name='Sales')

            # Convert Date column to datetime
            ro_data_melted['Date'] = pd.to_datetime(ro_data_melted['Date'], errors='coerce')

            # Add a date range selector
            start_date = st.date_input("Select start date")
            end_date = st.date_input("Select end date")

            # Convert user-selected dates to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Filter data based on the user-selected date range
            filtered_data = ro_data_melted[(ro_data_melted['Date'] >= start_date) & (ro_data_melted['Date'] <= end_date)]

            # Create a single plot with multiple lines
            fig = go.Figure()
            for ro in selected_ros:
                ro_filtered_data = filtered_data[filtered_data['RO Name'] == ro]
                fig.add_trace(go.Scatter(x=ro_filtered_data['Date'], y=ro_filtered_data['Sales'], mode='lines', name=ro))

            fig.update_layout(title=f"Sales Data for Selected ROs between {start_date.date()} and {end_date.date()}", height=600)
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == 'GA-wise':
        # GA-wise analysis
        ga_options = combined_df['GA'].unique().tolist()
        selected_gas = st.multiselect("Select GAs for analysis", ga_options)

        if not selected_gas:
            st.write("Please select at least one GA for analysis.")
        else:
            ga_data = combined_df[combined_df['GA'].isin(selected_gas)].copy()

            # Melt the dataframe
            ga_data_melted = ga_data.melt(id_vars=['GA', 'RO Name'], var_name='Date', value_name='Sales')

            # Convert Date column to datetime
            ga_data_melted['Date'] = pd.to_datetime(ga_data_melted['Date'], errors='coerce')

            # Add a date range selector
            start_date = st.date_input("Select start date")
            end_date = st.date_input("Select end date")

            # Convert user-selected dates to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Filter data based on the user-selected date range
            filtered_data = ga_data_melted[(ga_data_melted['Date'] >= start_date) & (ga_data_melted['Date'] <= end_date)]

            # Create a single plot with multiple bars
            fig = go.Figure()
            for ga in selected_gas:
                ga_filtered_data = filtered_data[filtered_data['GA'] == ga]
                fig.add_trace(go.Bar(x=ga_filtered_data['Date'], y=ga_filtered_data['Sales'], name=ga))

            fig.update_layout(title=f"Sales Data for Selected GAs between {start_date.date()} and {end_date.date()}", height=600, barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == 'XGBoost Prediction':
        # XGBoost prediction
        xgboost_prediction(combined_df)

    elif analysis_type == 'Chat with AI':
        # Chat with AI
        pandasai_chat(combined_df)
