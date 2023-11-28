import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import plotly.express as px
import xgboost as xgb
import joblib
import streamlit as st

# Load the data
df = pd.read_csv('Churn_Modelling.csv')

# Dropping Irrelevant Features
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encoding Categorical Data
df = pd.get_dummies(df, drop_first=True)

# Some insights about the target variable
X = df.drop('Exited', axis=1)
y = df['Exited']

# Handling Imbalanced Data with SMOTE
X_res, y_res = SMOTE().fit_resample(X, y)

# Splitting The Dataset into Training Set and Test Set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=47)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train)
y_pred1 = log.predict(X_test)

# SVC
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred2 = svm_model.predict(X_test)

# KNeighbors Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred3 = knn.predict(X_test)

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred4 = dt.predict(X_test)

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred5 = rf.predict(X_test)

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred6 = gbc.predict(X_test)

# XGBoost
model_xgb = xgb.XGBClassifier(random_state=42, verbosity=0)
model_xgb.fit(X_train, y_train)
y_pred7 = model_xgb.predict(X_test)

# Accuracy Summary
performance_summary = pd.DataFrame({
    'Model': ['LR', 'SVC', 'KNN', 'DT', 'RF', 'GBC', 'XGB'],
    'ACC': [accuracy_score(y_test, y_pred1),
            accuracy_score(y_test, y_pred2),
            accuracy_score(y_test, y_pred3),
            accuracy_score(y_test, y_pred4),
            accuracy_score(y_test, y_pred5),
            accuracy_score(y_test, y_pred6),
            accuracy_score(y_test, y_pred7)
            ],
    'PRECISION': [precision_score(y_test, y_pred1),
                  precision_score(y_test, y_pred2),
                  precision_score(y_test, y_pred3),
                  precision_score(y_test, y_pred4),
                  precision_score(y_test, y_pred5),
                  precision_score(y_test, y_pred6),
                  precision_score(y_test, y_pred7)
                  ]
})

# Saving the best model, XGBoost
model_xgb.fit(X_res, y_res)
joblib.dump(model_xgb, 'churn_predict_model')

# Load the saved model
model = joblib.load('churn_predict_model')

# Streamlit App
title_style = """
    <style>
        .title {
            font-size: 3em;
            color: #ffffff; /* White text color */
            background-color: #000000; /* Black background color */
            padding: 20px; /* Add padding for better visibility */
            border-radius: 10px; /* Add rounded corners */
            text-align: center; /* Center the text */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
            transition: background-color 0.3s ease; /* Add a transition effect for background color */
        }

        .title:hover {
            background-color: #c0c0c0; /* Silver color on hover */
        }
    </style>
"""

# Apply the custom style to the title
st.markdown(title_style, unsafe_allow_html=True)
st.markdown("<p class='title'>Churn Prediction Web App</p>", unsafe_allow_html=True)

image_style = """
    <style>
        .image-container {
            transition: transform 0.5s ease; /* Add a transition effect for the transform property */
        }

        .image-container:hover {
            transform: scale(1.5); /* Scale the image up by 10% on hover */
        }
    </style>
"""

# Apply the custom style to the image
st.markdown(image_style, unsafe_allow_html=True)

# Display the image with the specified path
image_path = "Cust3.gif"
st.image(image_path, caption="Stop the Customer", use_column_width=True, output_format='GIF')


# Sidebar
image_path2 = "Cust.jpg"
st.sidebar.image(image_path2,use_column_width=True)
st.sidebar.header("User Input Features")

# Convert user input to a pandas DataFrame
user_input_values = {}
for feature in X.columns:  # Use X.columns directly
    min_val = float(X[feature].min())  # Ensure min_val is a float
    max_val = float(X[feature].max())  # Ensure max_val is a float
    default_val = (min_val + max_val) / 2.0  # Ensure default_val is a float

    if X[feature].dtype == 'float64':
        # Use slider for float features
        user_input_value = st.sidebar.text_input(f"Enter {feature}:", float(default_val), key=f"{feature}_text")
    elif feature in ['HasCrCard','IsActiveMember','Geography_Germany', 'Geography_Spain', 'Gender_Male']:
        # Use slider for features with values 0 or 1
        user_input_value = st.sidebar.slider(f"Enter {feature}:", 0, 1, int(default_val), key=f"{feature}_slider")
    else:
        # Use text input for other non-float features (accepts only integer values)
        user_input_value = st.sidebar.text_input(f"Enter {feature}:", int(default_val), key=f"{feature}_text")

        # Validate input to ensure it is an integer
        try:
            user_input_value = int(user_input_value)
        except ValueError:
            st.sidebar.warning(f"Please enter a valid integer for {feature}. Resetting to default.")
            user_input_value = int(default_val)

    user_input_values[feature] = user_input_value

# Convert user input to a pandas DataFrame
user_input_df = pd.DataFrame([user_input_values], columns=X.columns)  # Use X.columns directly

# Ensure proper data types for the user input DataFrame
for col in user_input_df.columns:
    if user_input_df[col].dtype == 'object':
        # Convert object type to numeric (assuming they are numeric)
        user_input_df[col] = pd.to_numeric(user_input_df[col], errors='coerce')

# One-hot encode categorical variables
user_input_df = pd.get_dummies(user_input_df, drop_first=True)

# Styling for the user input DataFrame
formatted_user_input_df = user_input_df.style.format(
    {col: "{:.3f}" if user_input_df[col].dtype == 'float64' else "{}" for col in user_input_df.columns}
).set_properties(**{'background-color': 'lightyellow', 'color': 'black'})
st.sidebar.subheader("User Input DataFrame:")
st.sidebar.dataframe(formatted_user_input_df)

# Prediction button
if st.sidebar.button("Get Prediction"):
    # Predict on user input
    prediction = model.predict(user_input_df)

    # Determine background color based on the prediction
    bg_color = "#e74c3c" if prediction[0] == 1 else "#2ecc71"

    # Display the prediction result with simulated pop-up effect
    popup_id = "prediction-popup"
    st.sidebar.markdown(
        f"""
        <style>
            #{popup_id} {{
                background-color: {bg_color}; /* Use dynamic background color */
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }}
            .popup-text {{
                color: #ffffff; /* Change the text color to white or your preference */
            }}
            .close-btn {{
                color: #ffffff;
                cursor: pointer;
            }}
        </style>
        <div id="{popup_id}">
            <h2 class="popup-text">Prediction Result</h2>
            <p class="popup-text">Customer <span style="background-color: {bg_color}; color: #ffffff; padding: 2px; border-radius: 4px;">{'will Churn' if prediction[0] == 1 else 'will not churn'}</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())


# Dropping irrelevant features
st.subheader("Dataset Features")
selected_columns = st.multiselect("Select columns to display", X.columns.tolist(), default=X.columns.tolist())
st.dataframe(X[selected_columns])

# EDA Visualizations
subheader_style = """
    <style>
        .subheader {
            font-size: 2.5em;
            color: #ffffff; /* White text color */
            background-color: rgba(52, 152, 219, 0.5); /* More transparent modern blue background color */
            padding: 10px; /* Add padding for better visibility */
            border-radius: 8px; /* Add rounded corners */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
            text-align: center; /* Center the text */
            transition: background-color 0.3s ease; /* Add a transition effect for background color */
        }

        .subheader:hover {
            background-color: rgba(52, 152, 219, 0.8); /* Adjusted background color on hover */
        }
    </style>
"""

# Apply the custom style to the subheader
st.markdown(subheader_style, unsafe_allow_html=True)
st.markdown("<p class='subheader'>Exploratory Data Analysis (EDA)</p>", unsafe_allow_html=True)
image_path2 = "Cust4.jpg"
st.image(image_path2,use_column_width=True)

# Assuming df is your DataFrame
target_variable = 'Exited'  # Change this to your actual target variable column name

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Countplot for visualization
sns.countplot(x=target_variable, data=df, ax=ax)
ax.set_title(f"Distribution of Target Variable ({target_variable})")

# Convert countplot to pie chart
ax.clear()  # Clear the existing plot

# Get the counts of each category
counts = df[target_variable].value_counts()

# Define colors for the pie chart
colors = ['#3498db', '#e74c3c']

# Plotting the pie chart with enhanced features
ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
ax.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.

# Add a title to the pie chart
ax.set_title(f"Distribution of Target Variable ({target_variable})")

# Display the pie chart in Streamlit using Plotly
fig, ax = plt.subplots()
st.plotly_chart(px.pie(df, names=target_variable, title=f"Distribution of Target Variable ({target_variable})", color_discrete_sequence=colors).update_traces(
    textposition='inside', 
    textinfo='percent+label', 
    hoverinfo='label+percent',
    marker=dict(line=dict(color='#000000', width=2))
))


# Correlation Heatmap
st.subheader("Correlation Heatmap")
selected_features = st.multiselect("Select features:", df.columns)

# Filter the DataFrame based on user selection
selected_df = df[selected_features]

# Check if the selected DataFrame is not empty
if not selected_df.empty and not selected_df.dropna().empty:
    # Drop rows and columns with NaN values
    selected_df = selected_df.dropna()

    # Correlation Heatmap
    corr_matrix = selected_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.warning("No data available for the selected features.")


# Set style for a modern look
sns.set_theme(style="whitegrid")

# Distribution of Numerical Features
st.title("Distribution of Numerical Features")

# Select numerical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns

# Allow users to choose a feature from a dropdown
selected_feature = st.selectbox("Select a Numerical Feature", num_features)

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot distribution for the selected feature with a unique color
color_index = num_features.get_loc(selected_feature) % len(sns.color_palette())
sns.histplot(df[selected_feature], kde=True, color=sns.color_palette('pastel')[color_index], bins=30)

# Customize plot and layout
ax.set_title(f'Distribution of {selected_feature}', fontsize=16)
ax.set_xlabel(selected_feature, fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

# Add grid for a cleaner look
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot using Streamlit
st.pyplot(fig)

# Add additional information or insights below the plot
st.markdown(f"Additional insights about {selected_feature} can be added here.")



sns.set_theme(style="whitegrid")

# Boxplots for Numerical Features vs. Target Variable
st.subheader("Boxplots for Numerical Features vs. Target Variable")

# Create a figure with subplots
fig, axes = plt.subplots(nrows=len(num_features), ncols=1, figsize=(10, 2.5 * len(num_features)))

# Iterate through numerical features and create colorful boxplots
for i, feature in enumerate(num_features):
    # Choose a unique color for each boxplot
    color = sns.color_palette('pastel')[i % len(sns.color_palette())]

    sns.boxplot(x='Exited', y=feature, data=df, ax=axes[i], color=color)
    axes[i].set_title(f'{feature} vs. Exited', fontsize=14)
    axes[i].set_xlabel('Exited', fontsize=12)
    axes[i].set_ylabel(feature, fontsize=12)
    axes[i].tick_params(axis='both', which='major', labelsize=10)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
fig.tight_layout()

# Display the plot using Streamlit
st.pyplot(fig)


#Ai models
subheader_style = """
    <style>
        .subheader {
            font-size: 2.5em;
            color: #ffffff; /* White text color */
            background-color: rgba(255, 69, 0, 0.8); /* Orange background color with transparency */
            padding: 15px; /* Add padding for better visibility */
            border-radius: 8px; /* Add rounded corners */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
            text-align: center; /* Center the text */
            transition: background-color 0.3s ease; /* Add a transition effect for background color */
        }

        .subheader:hover {
            background-color: rgba(255, 69, 0, 1); /* Darker orange color on hover */
        }
    </style>
"""

# Apply the custom style to the subheader
st.markdown(subheader_style, unsafe_allow_html=True)
st.markdown("<p class='subheader'>Various Training Model And Selection</p>", unsafe_allow_html=True)
# List of models and their instances
models = [('LR', LogisticRegression()),
          ('SVC', SVC()),
          ('KNN', KNeighborsClassifier()),
          ('DT', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('GBC', GradientBoostingClassifier()),
          ('XGB', xgb.XGBClassifier(random_state=42, verbosity=0))]

# Define a list of unique color maps for each model
color_maps = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'RdYlBu', 'viridis']

# Set the number of columns for the grid layout
num_columns = 2

# Header styling
st.title("Model Evaluation Heatmaps")
st.markdown("---")

# Create a layout with columns
columns = st.columns(num_columns)

# Iterate over models and color maps
for i, (model_name, model_instance) in enumerate(models):
    # Train the model
    model_instance.fit(X_train, y_train)

    # Make predictions
    y_predict = model_instance.predict(X_test)

    # Display heatmap for confusion matrix with a unique color map for each model
    with columns[i % num_columns]:  # Switch to a new column after every `num_columns` models
        # Card-style layout for each heatmap
        st.markdown(f"## {model_name}")
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(8, 6))
        conf_matrix = confusion_matrix(y_test, y_predict)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=color_maps[i], cbar=False, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix - {model_name}')
        st.pyplot(fig)




# Display Accuracy Summary
st.subheader("Model Accuracy Summary")

# Apply styling to the DataFrame
styled_summary = performance_summary[['Model', 'ACC']].style \
    .bar(subset=['ACC'], color='#2ecc71', vmin=0, vmax=1) \
    .set_properties(**{'background-color': '#f5f5f5', 'color': 'black', 'border': '1px solid #ddd'})
# Display the styled DataFrame
st.table(styled_summary)


# Display Accuracy Bar Plot
st.subheader("Model Accuracy Bar Plot")

# Create a bar plot using Plotly Express
fig = px.bar(performance_summary, x='ACC', y='Model', orientation='h',
             text='ACC', labels={'ACC': 'Accuracy'}, color='ACC',
             color_continuous_scale='Viridis')

# Update layout for better presentation
fig.update_layout(
    title='Model Accuracy Comparison',
    xaxis_title='Accuracy',
    yaxis_title='Model',
    yaxis_categoryorder='total ascending',
    uniformtext_minsize=10,
    uniformtext_mode='hide',
)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)


# Display Precision Summary
st.subheader("Model Precision Summary")

# Apply styling to the DataFrame
styled_precision_summary = performance_summary[['Model', 'PRECISION']].style \
    .bar(subset=['PRECISION'], color='#3498db', vmin=0, vmax=1) \
    .set_properties(**{'background-color': '#f5f5f5', 'color': 'black', 'border': '1px solid #ddd'})
    
# Display the styled DataFrame
st.table(styled_precision_summary)

# Display Precision Bar Plot
st.subheader("Model Precision Bar Plot")

# Create a bar plot using Plotly Express
fig = px.bar(performance_summary, x='PRECISION', y='Model', orientation='h',
             text='PRECISION', labels={'PRECISION': 'Precision'}, color='PRECISION',
             color_continuous_scale='Viridis')

# Update layout for better presentation
fig.update_layout(
    title='Model Precision Comparison',
    xaxis_title='Precision',
    yaxis_title='Model',
    yaxis_categoryorder='total ascending',
    uniformtext_minsize=10,
    uniformtext_mode='hide',
)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)


#Display Performance summary
st.subheader("Model Performance Summary")
st.dataframe(performance_summary)

#Display Performance summary Bar plot
st.subheader("Model Performance Summary Bar plot")
st.bar_chart(performance_summary)



