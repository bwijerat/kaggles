
import pandas as pd


# Intro to Machine Learning
##########################################################


# save filepath to variable for easier access
melbourne_file_path = 'data/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
md = melbourne_data.describe()

# Exercise: Explore Your Data
##########################################################
avg_lot_size = md['Landsize']['mean']

current_year = 2021
newest_home_age = int(current_year - md['YearBuilt']['max'])




print()

