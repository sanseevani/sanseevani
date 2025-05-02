
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import folium

# Load and preprocess data
df = pd.read_csv('traffic_accidents.csv')
df.fillna(method='ffill', inplace=True)
df['Weather'] = LabelEncoder().fit_transform(df['Weather'])
df['Road_Type'] = LabelEncoder().fit_transform(df['Road_Type'])

# Feature selection and model training
X = df[['Weather', 'Road_Type', 'Speed_limit']]
y = df['Accident_Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier().fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Map accident hotspots
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)
for _, row in df.iterrows():
    folium.Circle([row['Latitude'], row['Longitude']], radius=30, color='red').add_to(m)
m.save("accident_hotspots.html")
