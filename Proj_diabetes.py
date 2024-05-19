# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:10:52 2024

@author: giera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import mysql.connector
from sqlalchemy import create_engine

file_path = 'C:/Users/giera/OneDrive/Dokumenty/Python_Scripts/Proj_Diabetes/Diabetes.csv'

#Wgranie pliku
data = pd.read_csv(file_path)
data_copy = data

#Sprawdzenie rozmiarów, inspekcja wyglądu danych
data.shape # 768 wierszy i 10 kolumn
data.dtypes # Dane należą albo do int albo float64 

data.head()
data.describe()

# Sprawdzenie czy nie występują wartoci NA w danych
data.isna().sum()

# Sprawdzenie czy w danych informujących o poziomie glukozy we krwii, BMI, Cisnieniu krwii, 
# grubosci skóry oraz wieku nie ma wartosci równych 0. W przypadku glukozy nie ma danych
# co do użytych jednostek więc użyję mg/dl


indexy_zero = {}

for col in data_copy.columns[:8]:
    potencial_zero = data_copy.query(f'{col} == 0')
    zero_list = list(potencial_zero.index)
    print(f'W kolumnie {col} występuje {len(zero_list)} wartosci równych zero')
    indexy_zero[col] = zero_list
    
# Analiza czy rozkład naszych danych jest normalny

for i in data_copy.columns[:8]:
    stat_val, p_val1 = stats.shapiro(data[i])
    print(f'Wartosc testu shapiro_wilka dla {i} = {stat_val}, a jego p_value = {p_val1}')


# Zmiana 0 w kolumnie BMI  na wartosci srednie
BMI_mean = data_copy['BMI'].mean().round()
data_copy.loc[data_copy['BMI'] == 0, 'BMI'] = BMI_mean


# Sprawdzenie korelacji każdego z czynników (kolumn) by zidentyfikowac ich wpływ na cukrzycę
# oraz sprawdzenie czy nasz wynik jest istotny statystycznie

# Analiza wykazała, że dane w każdej kolumnie NIE należą do rozkładu normalnego

p_val = 0.05

for factor in data_copy.columns[:8]:
    corr_val, p_val_factor = stats.pearsonr(data['Outcome'], data[factor])
    if p_val_factor < p_val:     
        print(f'Współczynnik korelacji Pearsona dla {factor} = {corr_val}, a jego p value = {p_val_factor} ')
    else:
        print(f'P value dla {factor} jest nieistotne statystycznie')

# Analiza wykazała brak korelacji między cinieniem krwii, a cukrzycą. Najwyższe współczynniki
# korelacji wykazały: Pregnancies, Glucose, BMI, Age
    

# Analiza za pomocą regresji liniowej wieloczynnikowej 2 czynników, które wykazały najwyższy wynik
# korelacji (BMI oraz Glucose):
    
# H0: Brak istotnego związku między BMI, Glukozą a wywołaniem cukrzycy.
# H1: Istnieje istotny związek między BMI, Glukozą a wywołaniem cukrzycy.

x1 = data_copy[['BMI', 'Glucose']]
y = data_copy['Outcome']

x = sm.add_constant(x1)
model_linear = sm.OLS(y, x)

result_linear = model_linear.fit()
print(result_linear.summary())

# Na podstawie uzyskanych wyników można stwierdzić, że zarówno BMI, jak i Glukoza 
# wykazują istotny wpływ na wywołanie cukrzycy. Zgodnie z tym wynikiem, odrzucamy 
# hipotezę zerową (H0) na rzecz hipotezy alternatywnej (H1).

# Analiza za pomocą regresji logarytmicznej oraz przewidywanie 
# Przewidywanie za pomocą regresji logarytmicznej czy badana osoba ma cukrzycę



x1_log = data_copy.iloc[:,:8]
y_log = data_copy['Outcome']

x_log = sm.add_constant(x1_log)
model_log = sm.Logit(y_log, x_log)
result_log = model_log.fit()
print(result_log.summary())

data['Predicted_Probability'] = result_log.predict(x_log)

# Dane z wynikami regresji logistycznej

czynniki = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
coef = [0.1232, 0.0352, -0.0133, 0.0006, -0.0012, 0.0897, 0.9452, 0.0149]
std_err = [0.032, 0.004, 0.005, 0.007, 0.001, 0.015, 0.299, 0.009]

# Plot
plt.figure(figsize=(10, 6))

# Współczynniki
plt.errorbar(coef, range(len(czynniki)), xerr=std_err, fmt='o', capsize=5)

# Ustawienia osi
plt.yticks(range(len(czynniki)), czynniki)
plt.xlabel('Współczynniki regresji')
plt.title('Wpływ czynników na wystąpienie cukrzycy (regresja logistyczna)')

plt.grid(True)
plt.show()

# Jak widać na wykresie, DiabetesPedigreeFunction ma największy wpływ na przewidywanie modelu 
# regresji związanej z wystąpieniem cukrzycy. 




# Transport danych do MySQL

mydb = mysql.connector.connect(
    host = 'hostname',
    username = 'root',
    password = 'moje_haslo',
    database = 'proj_database'
    )

mycursor = mydb.cursor()

#mycursor.execute('CREATE DATABASE proj_database')

# mycursor.execute('CREATE TABLE Diabetes (Pregnancies int,Glucose int,BloodPressure int,SkinThickness int,Insulin int,BMI int,DiabetesPedigreeFunction int,Age int,Outcome int,Predicted_Probability int)')
columns_sql = ', '.join([i + ' int' for i in data_copy.columns])

mycursor.execute(f'CREATE TABLE Diabetes ({columns_sql})')

# Przesłanie dataframe do SQL
engine  = create_engine('mysql+mysqlconnector://root:moje_haslo@hostname/proj_database')

data_sql = data_copy.to_sql('diabetes', con = engine, if_exists = 'append', index = True)

