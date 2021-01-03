# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler as mini
from sklearn.model_selection import train_test_split
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=pd.read_csv('c_force_data.csv')
#print('COLLECTEDFEATURESINCLUDEDINTHEDATASET')
X=data.drop(['rand. prices for AI'],axis=1)
y=data['rand. prices for AI']
# y=labelencoder.fit_transform(y)
# mini = mini()
# X=mini.fit(X)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.12)

# mini = mini()
# X = mini.fit(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.13)
from sklearn.linear_model import LinearRegression,LogisticRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('models/c-force_model.pkl','wb'))

# data = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_ltcusd.csv')
# df0,df1 = data.shape[0], data.shape[1]
# print('{} Dates '.format(df0))
# # data= data.drop(['Date'], axis =1)
# # data = data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small dataset, we will train our model with all availabe data.
# X= data.drop(['close'],axis=1)
# y= data['close']
# mini = mini()
# X = mini.fit_transform(X)

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.64)
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #Fitting model with trainig data
# regressor.fit(X_train, y_train)

# # Saving model to disk
# pickle.dump(regressor, open('models/lit_model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('models/lit_model.pkl','rb'))
# future_x = X
# # X = X[3295:3302]
# # future_x = X[-1]
# # x = X[:-1]
# bata = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_ltcusd.csv')
# date = bata['time']
# date = date.tail()
# print(date)
# bata = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_ltcusd.csv')
# date = bata['time']
# print('PREDICTED Close')
# y = model.predict(future_x)
# print(y[-1:])


# data = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_btcusd.csv')
# df0,df1 = data.shape[0], data.shape[1]
# print('{} Dates '.format(df0))
# # data= data.drop(['Date'], axis =1)
# # data = data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small dataset, we will train our model with all availabe data.
# X= data.drop(['close'],axis=1)
# y= data['close']

# X = mini.fit_transform(X)

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.64)
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #Fitting model with trainig data
# regressor.fit(X_train, y_train)

# # Saving model to disk
# pickle.dump(regressor, open('models/bit_model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('models/bit_model.pkl','rb'))
# future_x = X
# # X = X[3295:3302]
# # future_x = X[-1]
# # x = X[:-1]
# bata = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_btcusd.csv')
# date = bata['time']
# date = date.tail()
# print(date)
# bata = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_btcusd.csv')
# date = bata['time']
# print('PREDICTED Close')
# y = model.predict(future_x)
# print(y[-1:])


# eth_data = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_ethusd.csv')
# df0,df1 = eth_data.shape[0], eth_data.shape[1]
# print('{} dates'.format(df0))
# # eth_data= eth_data.drop(['Date'], axis =1)
# # eth_data = eth_data.drop('Adj Close',axis=1)
# eth_X= eth_data.drop(['close'],axis=1)
# eth_y= eth_data['close']
# eth_X = mini.fit_transform(eth_X)
# eth_X_train,eth_X_test,eth_y_train,eth_y_test = train_test_split(eth_X,eth_y,test_size=.64)
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #Fitting model with trainig data
# regressor.fit(eth_X_train, eth_y_train)

# # Saving model to disk
# pickle.dump(regressor, open('models/eth_model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('models/eth_model.pkl','rb'))
# eth_future_x = eth_X
# # eth_X = eth_X[-1:]
#     # future_x = X[-1]
#     # x = X[:-1]
# eth_bata = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_ethusd.csv')
# eth_date = eth_bata['time']
# eth_date = eth_date.tail()
# eth_bata = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_ethusd.csv')
# eth_date = eth_bata['time']
# print('PREDICTED Close')
# eth_y = model.predict(eth_future_x)
# print('accuracy {}'.format(model.score(eth_X_test,eth_y_test)))
# eth_output =eth_y[-1:]
# print(eth_output)

# xlm_data = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_xlmusd.csv')
# df0,df1 = xlm_data.shape[0], xlm_data.shape[1]
# print('{} dates'.format(df0))
# # xlm_data= xlm_data.drop(['Date'], axis =1)
# # xlm_data = xlm_data.drop('Adj Close',axis=1)
# xlm_X= xlm_data.drop(['close'],axis=1)
# xlm_y= xlm_data['close']
# xlm_X = mini.fit_transform(xlm_X)
# xlm_X_train,xlm_X_test,xlm_y_train,xlm_y_test = train_test_split(xlm_X,xlm_y,test_size=.64)
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #Fitting model with trainig data
# regressor.fit(xlm_X_train, xlm_y_train)

# # Saving model to disk
# pickle.dump(regressor, open('models/xlm_model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('models/xlm_model.pkl','rb'))
# xlm_future_x = xlm_X
# # xlm_X = xlm_X[-1:]
#     # future_x = X[-1]
#     # x = X[:-1]
# xlm_bata = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_xlmusd.csv')
# xlm_date = xlm_bata['time']
# xlm_date = xlm_date.tail()
# xlm_bata = pd.read_csv('data/crypto/crypto_portfolio/15m/bitfinex_xlmusd.csv')
# xlm_date = xlm_bata['time']
# print('PREDICTED Close')
# xlm_y = model.predict(xlm_future_x)
# print('accuracy {}'.format(model.score(xlm_X_test,xlm_y_test)))
# xlm_output =xlm_y[-1:]
# print(xlm_output)

# AAPL_data = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
# df0,df1 = AAPL_data.shape[0], AAPL_data.shape[1]
# print('{} dates'.format(df0))
# AAPL_data = AAPL_data.fillna(28.630752329973355)
# AAPL_data= AAPL_data.drop(['Date'], axis =1)
# AAPL_data = AAPL_data.drop('Adj Close',axis=1)
# AAPL_X= AAPL_data.drop(['Close'],axis=1)
# AAPL_y= AAPL_data['Close']
# AAPL_X = mini.fit_transform(AAPL_X)
# AAPL_X_train,AAPL_X_test,AAPL_y_train,AAPL_y_test = train_test_split(AAPL_X,AAPL_y,test_size=.64)
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #Fitting model with trainig data
# regressor.fit(AAPL_X_train, AAPL_y_train)

# # Saving model to disk
# pickle.dump(regressor, open('models/AAPL_model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('models/AAPL_model.pkl','rb'))
# AAPL_future_x = AAPL_X
# # AAPL_X = AAPL_X[9733:9740]
#     # future_x = X[-1]
#     # x = X[:-1]
# AAPL_bata = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
# AAPL_date = AAPL_bata['Date']
# AAPL_date = AAPL_date.tail()
# print(AAPL_date)
# AAPL_bata = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
# AAPL_date = AAPL_bata['Date']
# print('PREDICTED Close')
# AAPL_y = model.predict(AAPL_future_x)
# print(AAPL_y[-1:])
# AAPL_output =AAPL_y[-1:]


# MSFT_data = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
# df0,df1 = MSFT_data.shape[0], MSFT_data.shape[1]
# print('{} dates'.format(df0))
# MSFT_data= MSFT_data.drop(['Date'], axis =1)
# MSFT_data = MSFT_data.drop('Adj Close',axis=1)
# MSFT_X= MSFT_data.drop(['Close'],axis=1)
# MSFT_y= MSFT_data['Close']
# MSFT_y.mean()
# MSFT_X = mini.fit_transform(MSFT_X)
# MSFT_X_train,MSFT_X_test,MSFT_y_train,MSFT_y_test = train_test_split(MSFT_X,MSFT_y,test_size=.64)
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #Fitting model with trainig data
# regressor.fit(MSFT_X_train, MSFT_y_train)

# # Saving model to disk
# pickle.dump(regressor, open('models/MSFT_model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('models/MSFT_model.pkl','rb'))
# MSFT_future_x = MSFT_X
# # MSFT_X = MSFT_X[8407:8414]
#     # future_x = X[-1]
#     # x = X[:-1]
# MSFT_bata = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
# MSFT_date = MSFT_bata['Date']
# MSFT_date = MSFT_date.tail()
# print(MSFT_date)
# MSFT_bata = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
# MSFT_date = MSFT_bata['Date']
# print('PREDICTED Close')
# MSFT_y = model.predict(MSFT_future_x)
# print(MSFT_y[-1:])


# MSFT_output =MSFT_y[-1:]


# # # Importing the libraries
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import pickle
# # from sklearn.linear_model import LinearRegression
# # from sklearn.preprocessing import MinMaxScaler as mini
# # from sklearn.model_selection import train_test_split
# roku_data = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
# roku_df0,roku_df1 = roku_data.shape[0], roku_data.shape[1]
# print('{} dates'.format(roku_df0))
# roku_data= roku_data.drop(['Date'], axis =1)
# roku_data = roku_data.drop('Adj Close',axis=1)
# # #Splitting Training and Test Set
# # #Since we have a very small roku_dataset, we will train our model with all availabe roku_data.
# roku_X= roku_data.drop(['Close'],axis=1)
# roku_y= roku_data['Close']
# roku_X = mini.fit_transform(roku_X)
# # #
# roku_X_train,roku_X_test,roku_y_train,roku_y_test = train_test_split(roku_X,roku_y,test_size=.64)
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# #
# # #Fitting model with trainig roku_data
# regressor.fit(roku_X_train, roku_y_train)

# # Saving model to disk
# pickle.dump(regressor, open('models/roku_model.pkl','wb'))
# #
# # # Loading model to compare the results
# roku_model = pickle.load(open('models/roku_model.pkl','rb'))
# roku_future_x = roku_X
# # roku_X = roku_X[:]
# # future_x = X[-1]
# # x = X[:-1]
# roku_bata = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
# roku_date = roku_bata['Date']
# roku_date = roku_date.tail()
# roku_bata = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
# roku_date = roku_bata['Date']
# print(roku_date.tail())
# print('PREDICTED Close')
# roku_y = roku_model.predict(roku_future_x)
# print(roku_y[-1:])
