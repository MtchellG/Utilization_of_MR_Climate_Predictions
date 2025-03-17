# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:54:26 2025

@author: mitch
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'C:/Users/mitch/OneDrive/Documents/Kaggle Projects/weather_nasa/data/'

train_data = pd.read_csv(path + 'DailyDelhiClimateTrain.csv')

train_data['date'] = pd.to_datetime(train_data['date'])

#clean data a bit
pressure_mean = np.mean(train_data['meanpressure'])
pressure_std = np.std(train_data['meanpressure'])



for i in range(len(train_data)): 
    if train_data['meanpressure'][i] > (1030) or train_data['meanpressure'][i] < 990:
        print(train_data['meanpressure'][i], '\n')
        train_data['meanpressure'][i] = pressure_mean
        print(train_data['meanpressure'][i])


plt.figure(figsize = (12,9), dpi = 700)
plt.plot(train_data['date'], train_data['wind_speed'])
plt.show()





#correlation matrix
plt.figure(figsize = (12,9), dpi = 700)
plt.title('Correlation matrix')
sns.heatmap(train_data.corr())
plt.show()

print(train_data.corr())


window_size = 75

def movingaverage(data,window_size):
    movingaverage = []
    i = 0

    
    while i < len(data) - window_size + 1:
        
        window = data[i:i+ window_size]
        
        window_average = round(sum(window) / window_size, 2)
        
        movingaverage.append(window_average)
        
        i+=1
        
    return movingaverage
        
#smoothed data        
averagetemp = movingaverage(train_data['meantemp'],75) 
averagehum = movingaverage(train_data['humidity'],75)
averagewind = movingaverage(train_data['wind_speed'],75)   
averagepressure = movingaverage(train_data['meanpressure'],75)
 
date_moving = train_data['date'][window_size-1:] 

    
plt.figure(figsize=(16,9), dpi = 800)

plt.subplot(2,1,1)
plt.plot(date_moving, averagetemp, color = 'blue', label = 'Temperature($^\circ$C)')
plt.plot(date_moving, averagehum, color = 'red', label = 'Humidity(g/m$^3$)')
plt.plot(date_moving, averagewind, color = 'green', label = 'Windspeed(km/h)')
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend(loc = 'upper right')

plt.subplot(2,1,2)
plt.plot(date_moving, averagepressure, color = 'darkorange', label = 'Pressure(mbar)')
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()

plt.show()


def fourier(data):
    
    peak_frequencies = np.zeros(len(data))
    X = np.zeros(len(data), dtype=complex)
    frequency = np.zeros(len(data)) 

    
    for k in range(0,len(data)-1):
        
        for i in range(0, len(data)-1):
            X[k] += data[i] * np.exp( (1j * np.pi * 2 * k * i)  / int(len(data)) )
            if i == 0:
                X[0] = 0

        
        X[k] /= len(data)
        X[k] = np.sqrt(np.real(X[k])**2 + np.imag(X[k])**2)
        frequency[k] = k / len(data)
        
        
        
    for k in range(0,len(X)-1):
        
        
        
        if X[k] > X[k+1] and X[k] > X[k-1] and X[k] > 1:
            np.append(peak_frequencies, frequency[k])
            print(float(X[k]), frequency[k], 1/frequency[k])
        
    return X, frequency

print('\n\n Temperature Amplitude \n\n')
temp_amplitude, temp_frequency = fourier(train_data['meantemp'])
print('\n\n Humidity Amplitude \n\n')
hum_amplitude, hum_frequency = fourier(train_data['humidity'])
print('\n\n Wind Speed Amplitude \n\n')
wnspd_amplitude, wnspd_frequency = fourier(train_data['wind_speed'])
print('\n\n Pressure Amplitude \n\n')
pssur_amplitude, pssur_frequency = fourier(train_data['meanpressure'])


plt.figure(figsize=(16,9), dpi = 800)


plt.subplot(2,2,1)
plt.title('DFT Temperature')
plt.plot(temp_frequency[1:250], temp_amplitude[1:250], color = 'black')
plt.tick_params(axis='both', direction='in', top=True, right=True)


plt.subplot(2,2,2)
plt.title('DFT Humidity')
plt.plot(hum_frequency[1:250], hum_amplitude[1:250], color = 'black')
plt.tick_params(axis='both', direction='in', top=True, right=True)


plt.subplot(2,2,3)
plt.title('DFT Wind speed')
plt.plot(wnspd_frequency[1:250], wnspd_amplitude[1:250], color = 'black')
plt.tick_params(axis='both', direction='in', top=True, right=True)


plt.subplot(2,2,4)
plt.title('DFT Pressure')
plt.plot(pssur_frequency[1:250], pssur_amplitude[1:250], color = 'black')
plt.tick_params(axis='both', direction='in', top=True, right=True)



plt.show()


# Model testing



#%% Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



test_data = pd.read_csv(path + 'DailyDelhiClimateTest.csv')

test_data['date'] = pd.to_datetime(test_data['date'])

pressure_mean = np.mean(test_data['meanpressure'])

print(test_data['meanpressure'][0])
for i in range(len(test_data)):
    if test_data['meanpressure'][i] > (1030) or test_data['meanpressure'][i] < 990:
        print(test_data['meanpressure'][i], '\n')
        test_data['meanpressure'][i] = pressure_mean
        print(test_data['meanpressure'][i])


print(test_data['meanpressure'][0])


modelT = LinearRegression()

modelT.fit(train_data[['humidity', 'wind_speed', 'meanpressure']], train_data['meantemp'])

predictionsT = np.array(modelT.predict(test_data[['humidity', 'wind_speed', 'meanpressure']]))

errorT = np.zeros(len(test_data))

for i in range(len(test_data)):
    errorT[i] = (abs(predictionsT[i] - test_data['meantemp'][i])) / test_data['meantemp'][i]


modelWS = LinearRegression()

modelWS.fit(train_data[['meantemp','humidity', 'meanpressure']], train_data['wind_speed'])

predictionsWS = modelWS.predict(test_data[['meantemp', 'humidity', 'meanpressure']])

errorWS = np.zeros(len(test_data))

for i in range(len(test_data)):
    errorWS[i] = (abs(predictionsWS[i] - test_data['wind_speed'][i])) / test_data['wind_speed'][i]
    



modelH = LinearRegression()

modelH.fit(train_data[['meantemp','wind_speed', 'meanpressure']], train_data['humidity'])

predictionsH = modelH.predict(test_data[['meantemp','wind_speed', 'meanpressure']])

errorH = np.zeros(len(test_data))

for i in range(len(test_data)):
    errorH[i] = (abs(predictionsH[i] - test_data['humidity'][i])) / test_data['humidity'][i]


modelP = LinearRegression()

modelP.fit(train_data[['meantemp','humidity', 'wind_speed']], train_data['meanpressure'])

predictionsP = modelP.predict(test_data[['meantemp','humidity', 'wind_speed']])

errorP = np.zeros(len(test_data))

for i in range(len(test_data)):
    errorP[i] = (abs(predictionsP[i] - test_data['meanpressure'][i])) / test_data['meanpressure'][i]
    





celsius = r'($^\circ$C)'        
humidity = r'(g/m$^3$)'         
pressure = r'(mbar)'             
wind_speed = r'(km/h)'          
relative_error = r'($\varepsilon$)'

plt.figure(figsize = (31.5,12), dpi = 1000)
plt.title('Linear Regression')
plt.subplot(2,4,1)
plt.plot(test_data['date'], test_data['meantemp'], color = 'blue', label = 'Actual')
plt.title('Predicted Temperature' + celsius )
plt.xticks(plt.xticks()[0][::2])
plt.plot(test_data['date'], predictionsT, color = 'r',linewidth = 4, label = 'Predicted')
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()

plt.subplot(2,4,2)
plt.plot(test_data['date'], test_data['wind_speed'], color = 'blue', label = 'Actual')
plt.title('Predicted Wind Speed' + wind_speed)
plt.xticks(plt.xticks()[0][::2])
plt.plot(test_data['date'], predictionsWS, color = 'r',linewidth = 4, label = 'Predicted')
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()

plt.subplot(2,4,3)
plt.plot(test_data['date'], test_data['humidity'], color = 'blue', label = 'Actual')
plt.title('Humidity' + humidity)
plt.xticks(plt.xticks()[0][::2])
plt.plot(test_data['date'], predictionsH, color = 'r',linewidth = 4, label = 'Predicted')
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()

plt.subplot(2,4,4)
plt.plot(test_data['date'], test_data['meanpressure'], color = 'blue', label = 'Actual')
plt.title('Pressure' + pressure)
plt.xticks(plt.xticks()[0][::2])
plt.plot(test_data['date'], predictionsP, color = 'r',linewidth = 4, label = 'Predicted')
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()

plt.subplot(2,4,5)
plt.plot(test_data['date'], errorT, marker = 'o', linestyle = '--', color = 'black', label = 'error')
plt.xticks(plt.xticks()[0][::2])
plt.title('Relative Error'+ relative_error)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()

plt.subplot(2,4,6)
plt.plot(test_data['date'], errorWS, marker = 'o', linestyle = '--', color = 'black', label = 'error')
plt.xticks(plt.xticks()[0][::2])
plt.title('Relative Error'+ relative_error)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()

plt.subplot(2,4,7)
plt.plot(test_data['date'], errorWS, marker = 'o', linestyle = '--', color = 'black', label = 'error')
plt.xticks(plt.xticks()[0][::2])
plt.title('Relative Error'+ relative_error)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()

plt.subplot(2,4,8)
plt.plot(test_data['date'], errorP, marker = 'o', linestyle = '--', color = 'black', label = 'error')
plt.xticks(plt.xticks()[0][::2])
plt.title('Relative Error'+ relative_error)
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.legend()


plt.tight_layout()

plt.show()

print('Temperature prediction stats:')
r2 = r2_score(test_data['meantemp'], predictionsT)
mae = mean_absolute_error(test_data['meantemp'], predictionsT)
mse = mean_squared_error(test_data['meantemp'], predictionsT)
rmse = np.sqrt(mse)  


print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Coefficients: {modelT.coef_}")
print(f"Intercept: {modelT.intercept_}")


print('\n\n\n')
del r2
del mae
del mse
del rmse

print('Windspeed prediction stats:')
r2 = r2_score(test_data['wind_speed'], predictionsWS)
mae = mean_absolute_error(test_data['wind_speed'], predictionsWS)
mse = mean_squared_error(test_data['wind_speed'], predictionsWS)
rmse = np.sqrt(mse)  


print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Coefficients: {modelWS.coef_}")
print(f"Intercept: {modelWS.intercept_}")
print('\n\n\n')
del r2
del mae
del mse
del rmse

print('Humidity prediction stats:')
r2 = r2_score(test_data['humidity'], predictionsH)
mae = mean_absolute_error(test_data['humidity'], predictionsH)
mse = mean_squared_error(test_data['humidity'], predictionsH)
rmse = np.sqrt(mse)  


print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Coefficients: {modelH.coef_}")
print(f"Intercept: {modelH.intercept_}")
print('\n\n\n')
del r2
del mae
del mse
del rmse

print('Pressure prediction stats:')
r2 = r2_score(test_data['meanpressure'], predictionsP)
mae = mean_absolute_error(test_data['meanpressure'], predictionsP)
mse = mean_squared_error(test_data['meanpressure'], predictionsP)
rmse = np.sqrt(mse)  


print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Coefficients: {modelP.coef_}")
print(f"Intercept: {modelP.intercept_}")
print('\n\n\n')
del r2
del mae
del mse
del rmse

#%% F-statistic
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


k1 = 3
k2 = 0
y1 = predictionsWS
y2 = np.mean(test_data['wind_speed'])
y = test_data['wind_speed']
rss1 = 0
rss2 = 0
n = len(test_data)
for i in range(0,n):
    rss1 +=  (y[i] -y1[i]) ** 2
    rss2 += (y[i] - y2) ** 2
    
sigma1 =  rss1 - rss2
sigma2 = rss2/ (n - k2)

F = sigma1 / sigma2

print(f'F = {F}')

    
    
    




from scipy.stats import f

p = 3
N = len(test_data)


df1 = p
df2 = N - p - 1

print(df1)
print(df2)


a = 0.05

Fc = f.ppf(1 - a, df1, df2)

print(f"F-critical value: {Fc}")


#%%




import numpy as np


y_true = test_data['wind_speed'].values

rss = 0
tss = 0

for i in range(0,len(test_data)):
    rss += (y_true[i] - predictionsWS[i]) ** 2
    

# Compute total sum of squares (TSS)


for i in range(0,len(test_data)):
    tss += (y_true[i] - np.mean(y_true)) ** 2


print(np.mean(predictionsWS))
print(np.mean(y_true))
print(tss)





r2 = 1 - (rss / tss)

print("Manual R²:", r2)




#%%


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

k1 = 4
k2 = 1
y1 = predictionsH
y2 = np.mean(test_data['humidity'])
y = test_data['humidity']

n = len(test_data)

sigma1 = np.var(y2)
sigma2 = np.var(y1)

F = sigma1 / sigma2

print(f'F = {F}')




d1 = k1 - k2
d2 = n - k1


p_value = 1 - f.cdf(F, d1, d2)
'''
print(f'F = {F}')


print(f'F = {F}, p-value = {p_value}')
'''
x = np.linspace(F - 30, F + 30, 500)


plt.figure(figsize = (8,8), dpi = 700)
plt.plot(x, f.pdf(x, d1, d2), c = 'black')

plt.fill_between(x, f.pdf(x, d1, d2), where=(x >= F), color = 'r', alpha = 0.5)
plt.axvline(F, color='r', linestyle='--', label=f'F = {F:.2f}')
plt.title('F-distribution PDF (d1={d1}, d2={d2}')
plt.xlabel('F', fontsize= 15)
plt.ylabel(r"$\rho_P$", rotation = 0, fontsize = 15)
plt.legend()
plt.tick_params(axis='both', direction='in', top=True, right=True)
plt.show()

