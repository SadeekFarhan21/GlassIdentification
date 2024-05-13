# Import the libraries
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
# Create the dataframe
glass_identification = fetch_ucirepo(id=42) 
df = pd.DataFrame(glass_identification.data.features)
df['target'] = glass_identification.data.targets

X = df.drop('target', axis='columns')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=4)
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)

# Title and introduction
st.title('Glass Classification Tutorial')
st.write("""
In this tutorial, we will analyze the Glass Identification dataset and build a machine learning model to classify glass types. 
""")

# Create the dataframe
glass_identification = fetch_ucirepo(id=42) 
df = pd.DataFrame(glass_identification.data.features)
df['target'] = glass_identification.data.targets

# Show dataset information
st.header('Dataset Information')
st.write("Number of samples:", df.shape[0])
st.write("Number of features:", df.shape[1] - 1)  # Exclude target column
# Define feature explanations
feature_explanations = {
    'RI': 'Refractive Index of the glass',
    'Na': '**Percentage of Sodium** in the material',
    'Mg': '**Percentage of Magnesium** in the material',
    'Al': '**Percentage of Aluminum** in the material',
    'Si': '**Percentage of Silicon** in the material',
    'K': '**Percentage of Potassium** in the material',
    'Ca': '**Percentage of Calcium** in the material',
    'Ba': '**Percentage of Barium** in the material',
    'Fe': '**Percentage of Iron** in the material'
}

# Show dataset information with feature explanations
st.header('Dataset Information')
st.write("Number of samples:", df.shape[0])
st.write("Number of features:", df.shape[1] - 1)  # Exclude target column
st.write("Features and Explanations:")
for feature in df.columns[:-1]:
    st.write(f"- **{feature}:** {feature_explanations.get(feature, 'No explanation available')}")

st.write("Target variable:", 'target')

# Descriptive Statistics
st.header('Descriptive Statistics')
st.code('df.describe()', language='python', line_numbers=True)
st.dataframe(df.describe())

# Sample data
st.header('Sample Data')
st.dataframe(df.sample(n=8))

# Code for data analysis
st.header('Data Analysis Code')
st.code('''
# Import the libraries
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create the dataframe
glass_identification = fetch_ucirepo(id=42) 
df = pd.DataFrame(glass_identification.data.features)
df['target'] = glass_identification.data.targets

X = df.drop('target', axis='columns')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=4)
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
''', language='python', line_numbers=True)

# Sidebar for input values
st.sidebar.title('Glass Classification Input')
refractive_index = st.sidebar.slider('Refractive Index', 
                                     df['RI'].min(), df['RI'].max(), 
                                     step=0.001, value=df['RI'].median(), 
                                     format="%.3f")
sodium = st.sidebar.slider("% of Sodium", 
                            df['Na'].min(), df['Na'].max(), 
                            step=0.001, value=df['Na'].median(), 
                            format="%.3f")
magnesium = st.sidebar.slider("% of Magnesium", 
                               df['Mg'].min(), df['Mg'].max(), 
                               step=0.001, value=df['Mg'].median(), 
                               format="%.3f")
aluminum = st.sidebar.slider("% of Aluminum", 
                              df['Al'].min(), df['Al'].max(), 
                              step=0.001, value=df['Al'].median(), 
                              format="%.3f")
silicon = st.sidebar.slider("% of Silicon", 
                             df['Si'].min(), df['Si'].max(), 
                             step=0.001, value=df['Si'].median(), 
                             format="%.3f")
potassium = st.sidebar.slider("% of Potassium", 
                               df['K'].min(), df['K'].max(), 
                               step=0.001, value=df['K'].median(), 
                               format="%.3f")
calcium = st.sidebar.slider("% of Calcium", 
                             df['K'].min(), df['K'].max(), 
                             step=0.001, value=df['K'].median(), 
                             format="%.3f")
barium = st.sidebar.slider("% of Barium", 
                            df['Ba'].min(), df['Ba'].max(), 
                            step=0.001, value=df['Ba'].median(), 
                            format="%.3f")
iron = st.sidebar.slider("% of Iron", 
                          df['Fe'].min(), df['Fe'].max(), 
                          step=0.001, value=df['Fe'].median(), 
                          format="%.3f")

# Make prediction
# Make prediction
predicted_glass_type = rfc.predict([[refractive_index, sodium, magnesium, aluminum, silicon, potassium, calcium, barium, iron]])[0]

# Display prediction
st.subheader('Prediction')
st.write(f"The predicted glass type based on the input features is: **{predicted_glass_type}**")

# Provide additional information about the glass type
glass_type_description = {
    1: "Building Windows Float Processed",
    2: "Building Windows Non-Float Processed",
    3: "Vehicle Windows Float Processed",
    4: "Vehicle Windows Non-Float Processed",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}

if predicted_glass_type in glass_type_description:
    st.write(f"**Description:** {glass_type_description[predicted_glass_type]}")
else:
    st.write("No description available for this glass type.")
