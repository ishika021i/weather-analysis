from django.shortcuts import render

import requests #This project helps us to fetch data frtom API
import pandas as pd #for handling and analyzing data
import numpy as np #for numerical operations
from sklearn.model_selection import train_test_split #to split data into testing and training sets
from sklearn.preprocessing import LabelEncoder #to convert categorical data into numerical values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #models for classificaion and regression task
from sklearn.metrics import mean_squared_error #to measure the accuracy of our predictions
from datetime import datetime, timedelta #to handle date and time
import pytz
import os

API_KEY= '9b04487aee4422b81dd23a035df76994' #replace with your actual API key
BASE_URL= 'https://api.openweathermap.org/data/2.5/' #base url for making api requests

#fetch current weather data 
def get_current_weather(city):
    url=f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric" #construct the API request url
    response=requests.get(url) #send the get request to API
    data=response.json()
    return{
        'city':data['name'],
        'current_temp':round(data['main']['temp']),
        'feels_like':round(data['main']['feels_like']),
        'temp_min':round(data['main']['temp_min']),
        'temp_max':round(data['main']['temp_max']),
        'humidity':round(data['main']['humidity']),
        'description':data['weather'][0]['description'],
        'country':data['sys']['country'],
        'wind_gust_dir':data['wind']['deg'],
        'pressure':data['main']['pressure'],
        'Wind_Gust_Speed':data['wind']['speed'],

        'clouds': data['clouds']['all'],
        'visibility': data['visibility'],
    }

#read historical data
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df=df.dropna()
    df =df.drop_duplicates()
    return df
#prepare data for rain prediction model
def prepare_data(data):
    le= LabelEncoder()#labelencoder instance
    data['WindGustDir'] =le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] =le.fit_transform(data['RainTomorrow'])

    #define the feature variable and target variables
    x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']] #feature variables
    y =data['RainTomorrow'] #target variable
    return x, y, le #return feture variable, target variable and the label encoder

#Train Rain Prediction Model
from numpy.random.mtrand import rand
def train_rain_model(x,y):
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print("Mean squared Error for Rain model")
    print(mean_squared_error(y_test,y_pred))
    return model

#prepare regression data
def prepare_regression_data(data, feature):
    X, y = [], [] #initialize list for feature and target values
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])   # previous value
        y.append(data[feature].iloc[i+1]) # next value
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

#train regression model
def train_regression_model(x, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)
    return model

#predict future
def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
      next_value =model.predict(np.array([[predictions[-1]]]))
      predictions.append(next_value[0])
    return predictions[1:]

#weather analysis function
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')

        # ===================== CURRENT WEATHER =====================
        current_weather = get_current_weather(city)

        # ===================== LOAD HISTORICAL DATA =====================
        csv_path = r'C:\Users\jagdish\Desktop\WeatherAppProject\weather.csv'
        historical_data = read_historical_data(csv_path)

        # ===================== TRAIN RAIN MODEL =====================
        x, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(x, y)

        # ===================== WIND DIRECTION (SAFE) =====================
        wind_deg = current_weather.get('wind_gust_dir')

        if wind_deg is None:
            compass_direction = 'N'
        else:
            wind_deg = wind_deg % 360
            compass_points = [
                ('N', 0, 11.25), ('NNE', 11.25, 33.75), ('NE', 33.75, 56.25),
                ('ENE', 56.25, 78.75), ('E', 78.75, 101.25), ('ESE', 101.25, 123.75),
                ('SE', 123.75, 146.25), ('SSE', 146.25, 168.75), ('S', 168.75, 191.25),
                ('SSW', 191.25, 213.75), ('SW', 213.75, 236.25),
                ('WSW', 236.25, 258.75), ('W', 258.75, 281.25),
                ('WNW', 281.25, 303.75), ('NW', 303.75, 326.25),
                ('NNW', 326.25, 348.75), ('N', 348.75, 360)
            ]

            compass_direction = next(
                (point for point, start, end in compass_points
                 if start <= wind_deg < end),
                'N'
            )

        compass_direction_encoded = (
            le.transform([compass_direction])[0]
            if compass_direction in le.classes_
            else -1
        )

        # ===================== CURRENT DATA FOR MODEL =====================
        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['Wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp'],
        }

        current_df = pd.DataFrame([current_data])

        # ===================== RAIN PREDICTION =====================
        rain_prediction = rain_model.predict(current_df)[0]

        # ===================== REGRESSION MODELS =====================
        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

        temp_model = train_regression_model(x_temp, y_temp)
        hum_model = train_regression_model(x_hum, y_hum)

        future_temp = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])

        # ===================== TIME CALCULATION =====================
        timezone = pytz.timezone('Asia/Karachi')
        now = datetime.now(timezone)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        future_time = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
        time1, time2, time3, time4, time5 = future_time

        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity

        # ===================== TEMPLATE CONTEXT =====================
        context = {
            'location': city,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],
            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %Y"),
            'wind': current_weather['Wind_Gust_Speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],

            'time1': time1,
            'time2': time2,
            'time3': time3,
            'time4': time4,
            'time5': time5,

            'temp1': f"{round(temp1, 1)}",
            'temp2': f"{round(temp2, 1)}",
            'temp3': f"{round(temp3, 1)}",
            'temp4': f"{round(temp4, 1)}",
            'temp5': f"{round(temp5, 1)}",

            'hum1': f"{round(hum1, 1)}",
            'hum2': f"{round(hum2, 1)}",
            'hum3': f"{round(hum3, 1)}",
            'hum4': f"{round(hum4, 1)}",
            'hum5': f"{round(hum5, 1)}",
        }

        return render(request, 'weather.html', context)

    return render(request, 'weather.html')
