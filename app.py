import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# 加载模型
model_path = 'random_forest_model.pkl'
with open(model_path, 'rb') as file:  # 使用 pickle 加载模型
    model = pickle.load(file)

def predict_storm(features):
    prediction = model.predict([features])
    return prediction[0]

def evaluate_decision(storm_prediction, botrytis_chance, sugar_levels):
    if storm_prediction == 1:
        if botrytis_chance > 0.1:  # 示例阈值
            return ("High Risk of Storm with Botrytis", "Wait for Noble Rot")
        else:
            return ("High Risk of Storm, No Botrytis", "Harvest Now")
    else:
        if sugar_levels['high'] > 0.3:  # 示例阈值
            return ("Low Risk of Storm, High Sugar", "Wait for More Sugar")
        elif sugar_levels['typical'] > 0.3:
            return ("Low Risk of Storm, Typical Sugar", "Consider Gradual Harvest")
        else:
            return ("Low Risk of Storm, No Sugar", "Harvest Now")

# Streamlit 页面设置
st.title('Vineyard Decision Support Tool')
st.write('Use this tool to determine the optimal time to harvest grapes based on weather conditions.')

# 用户输入
weekly_precipitation = st.number_input('Weekly Precipitation (inches)', min_value=0.0, step=0.01, value=0.35)
max_temperature = st.number_input('Maximum Temperature (°F)', min_value=50, max_value=100, step=1, value=75)
min_temperature = st.number_input('Minimum Temperature (°F)', min_value=30, max_value=70, step=1, value=50)
days_with_rain = st.number_input('Days with Rain (days)', min_value=0, max_value=7, step=1, value=3)
storm_t1 = st.selectbox('Was there a storm last week?', ('Yes', 'No'), index=1)
storm_t2 = st.selectbox('Was there a storm two weeks ago?', ('Yes', 'No'), index=1)

# 贵腐菌和糖分水平可能性调整
botrytis_chance = st.slider('Chance of Botrytis (%)', 0, 100, 10) / 100.0
sugar_levels = {
    'high': st.slider('Chance of High Sugar Level (%)', 0, 100, 10) / 100.0,
    'typical': st.slider('Chance of Typical Sugar Level (%)', 0, 100, 30) / 100.0,
    'low': st.slider('Chance of Low Sugar Level (%)', 0, 100, 60) / 100.0
}

# 预测按钮
if st.button('Predict and Evaluate'):
    features = [weekly_precipitation, max_temperature, min_temperature, days_with_rain, int(storm_t1 == 'Yes'), int(storm_t2 == 'Yes')]
    storm_status = predict_storm(features)
    decision, recommendation = evaluate_decision(storm_status, botrytis_chance, sugar_levels)
    st.write(f'Prediction for next week: {"Storm" if storm_status else "No Storm"}')
    st.write(f'Decision e-value: {decision}')
    st.write(f'Recommended action: {recommendation}')
