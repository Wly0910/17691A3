import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# 数据预处理


def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['WEEK_OF_YEAR'] = df['DATE'].dt.isocalendar().week
    df_filtered = df[(df['WEEK_OF_YEAR'] >= 35) & (df['WEEK_OF_YEAR'] <= 40)]

    weekly_data = df_filtered.groupby([df_filtered['DATE'].dt.year, 'WEEK_OF_YEAR']).agg(
        weekly_precipitation=pd.NamedAgg(column='PRCP', aggfunc='sum'),
        max_temperature=pd.NamedAgg(column='TMAX', aggfunc='mean'),
        min_temperature=pd.NamedAgg(column='TMIN', aggfunc='mean'),
        days_with_rain=pd.NamedAgg(column='RAIN', aggfunc='sum')
    ).reset_index()

    weekly_data['STORM'] = weekly_data.apply(
        lambda x: (x['weekly_precipitation'] >= 0.35) and (
            x['max_temperature'] <= 80),
        axis=1
    )

    # Create temporal features for the model
    weekly_data['STORM_t-1'] = weekly_data['STORM'].shift(1)
    weekly_data['STORM_t-2'] = weekly_data['STORM'].shift(2)

    return weekly_data.dropna(subset=['STORM_t-1', 'STORM_t-2'])

# 模型训练


def train_model(data):
    X = data[['weekly_precipitation', 'max_temperature',
              'min_temperature', 'days_with_rain', 'STORM_t-1', 'STORM_t-2']]
    y = data['STORM']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 模型评估
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}')

    return model


if __name__ == "__main__":
    data_path = 'vineyard_weather_1948-2017.csv'
    processed_data = preprocess_data(data_path)
    model = train_model(processed_data)
    joblib.dump(model, 'random_forest_model.pkl')
    print("Model trained and saved successfully.")
