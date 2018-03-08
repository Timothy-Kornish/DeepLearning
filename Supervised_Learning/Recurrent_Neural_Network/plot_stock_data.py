# Recurrent Neural Network

# Loading in Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load in Data
stock = pd.read_csv('Google_Stock_Price_Train.csv')

print(stock.head())
print(stock.describe())
