# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:10:52 2024

@author: giera
"""

import pandas as pd
import numpy as np

file_path = 'C:/Users/giera/OneDrive/Dokumenty/Python_Scripts/Proj_Diabetes/Diabetes.csv'

data = pd.read_csv(file_path)

data.shape

data.head()

data.describe()

data.isna().sum()

