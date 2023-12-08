import pandas as pd
import json

cities = ['london', 'liverpool', 'bath']


columns = ['temp_' + city for city in cities] + \
          ['temp_min_' + city for city in cities] + \
          ['temp_max_' + city for city in cities] + \
          ['humidity_' + city for city in cities]
df = pd.DataFrame(columns=columns, index=[0])


for city in cities:
    with open(f'data/custom/current_forecast_{city}.json', 'r') as file:
        data = json.load(file)

    # Data Processing
    first_item = data['list'][0]['main']
    temp = first_item['temp']
    temp_min = first_item['temp_min']
    temp_max = first_item['temp_max']
    humidity = first_item['humidity']

    # Assign values to the DataFrame
    df[f'temp_{city}'] = temp
    df[f'temp_min_{city}'] = temp_min
    df[f'temp_max_{city}'] = temp_max
    df[f'humidity_{city}'] = humidity


df.to_csv('data/custom/current_forecast_combined.csv')

print(df)

# check the unix timestamp to see if this is current data 