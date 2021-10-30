
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
test = pd.read_csv('Data Set//test.csv')
item_categories = pd.read_csv('Data Set//item_categories.csv')
items = pd.read_csv('Data Set//items.csv')
shops = pd.read_csv('Data Set//shops.csv')
sales = pd.read_csv('Data Set/sales_train.csv',  parse_dates=['date'], infer_datetime_format=True, dayfirst=True)


print(sales.head())
print('____________________________')
print(sales.info())
print('____________________________')
print(sales.describe())

print(test.head())
print('____________________________')
print(test.info())
print('____________________________')
print(test.describe())

print(items.head())
print('____________________________')
print(items.info())
print('____________________________')
print(items.describe())

print(shops.head())
print('____________________________')
print(shops.info())
print('____________________________')
print(shops.describe())

print(item_categories.head())
print('____________________________')
print(item_categories.info())
print('____________________________')
print(item_categories.describe())

print(shops.head())
print('____________________________')
print(shops.info())
print('____________________________')
print(shops.describe())

df_item=pd.merge(items,item_categories,on='item_category_id',how='inner')
sales_train=pd.merge(sales,shops,on='shop_id',how='inner')
sales=pd.merge(sales_train,df_item,on='item_id',how='inner')

sales = sales[sales['shop_id'].isin(test['shop_id'].unique())]
sales = sales[sales['item_id'].isin(test['item_id'].unique())]

print(sales)

import seaborn as sns
print(sns.boxplot(x=sales.item_cnt_day))

import seaborn as sns
sns.boxplot(x=sales.item_price)

sales = sales[(sales.item_price < 300000 )& (sales.item_cnt_day < 1000)]
# remove negative item price
sales = sales[sales.item_price > 0].reset_index(drop = True)

sales = sales.groupby(["date_block_num","shop_id","item_id"])[['date_block_num','date', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']].agg({"date_block_num":'mean',"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

print(sales)
sales = sales.item_cnt_day.apply(list).reset_index()
print(sales)

sales_data = pd.merge(test,sales,on = ['item_id','shop_id'],how = 'left')

sales_data.fillna(0,inplace = True)
sales_data.drop(['shop_id','item_id'],inplace = True, axis = 1)

print(sales_data)
sales_data = sales_data.pivot_table(index = 'ID', columns='date_block_num', values = 'sum', aggfunc='sum')
print(sales_data)

sales_data = sales_data.fillna(0)
print(sales_data.head(20))

X=sales_data[sales_data.columns[:-1]]
y=sales_data[sales_data.columns[-1]]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, y, test_size=0.20, random_state=1)

import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.models import load_model, Model

# defining model
model = Sequential()
model.add(LSTM(units = 128,return_sequences=True,input_shape = (33,1)))
model.add(Dropout(0.5))
model.add(LSTM(units = 64,return_sequences=False,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(1))
# opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss = 'mse',optimizer = 'Nadam', metrics = ['mean_squared_error'])
print(model.summary())

