import numpy as np
from flask import Flask, session,abort,request, jsonify, render_template,redirect,url_for,flash
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as mini
import os
import stripe
import datetime

app = Flask(__name__)
pub_key ='pk_live_2pO0yUvt9xKyjAo9rca8Vkc600FWtgJuqZ'
lit_model = pickle.load(open('models/lit_model.pkl', 'rb'))
xlm_model = pickle.load(open('models/xlm_model.pkl', 'rb'))
bit_model = pickle.load(open('models/bit_model.pkl', 'rb'))
eth_model = pickle.load(open('models/eth_model.pkl', 'rb'))
AAPL_model = pickle.load(open('models/AAPL_model.pkl', 'rb'))
MSFT_model = pickle.load(open('models/MSFT_model.pkl', 'rb'))
ROKU_model = pickle.load(open('models/roku_model.pkl', 'rb'))
c_force_model = pickle.load(open('models/c-force_model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

def index():
    return render_template('index.html',pub_key=pub_key)
@app.route('/crypto')
def crypto():
    return render_template('crypto.html',pub_key=pub_key)

@app.route('/stocks')
def stocks():
    return render_template('stocks.html',pub_key=pub_key)
@app.route('/c_force')
def c_force():
    return render_template('c_force.html',pub_key=pub_key)
@app.route('/pay',methods=['POST'])
def pay():
    customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])
    charge = stripe.Charge.create(
        customer=customer.id,
        amount=19900,
        currency='usd',
        description='The Product'
    )


@app.route('/predict_c_force',methods=['POST'])
def predict_c_force():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    bid_values = np.random.uniform(0.015,0.06,[6000,1])
    bid_df = pd.DataFrame(bid_values,columns=['Bid'])
    stake_values = np.random.uniform(0.025,0.032,[6000,1])
    stake_df = pd.DataFrame(stake_values,columns=['Stake'])

    clean_values = np.random.uniform(0.015,0.025,[6000,1])
    clean_df = pd.DataFrame(clean_values,columns=['Deck1'])

    nsfw_values = np.random.uniform(0.026,0.04,[6000,1])  
    nsfw_df = pd.DataFrame(nsfw_values,columns=['Deck2']) 

    damage1_values = np.random.randint(0,3,[6000,1]) 
    damage_df = pd.DataFrame(damage1_values,columns=['Deck1_damage'])

    damage2_values = np.random.randint(3,7,[6000,1]) 
    damage2_df = pd.DataFrame(damage2_values,columns=['Deck2_damage'])    
    data = clean_df.join(nsfw_df)
    # data = data.join(full_df)
    data = data.join(damage_df) 
    data = data.join(damage2_df) 
    # data = data.join(damage3_df)
    data[:5]
    # data.to_csv('data/c_force_data.csv')  
    import os 
    # PATH FOR PRICE-DECK-TRADE
    # clean_path = os.listdir("../../dAIsy21App/lfr_api/media/media/anihotime/clean/")
    # clean_path = np.array(clean_path)
    # clean_path=pd.DataFrame(clean_path,columns=['Deck1 cards'])[:25] 
    # data = data.join(clean_path)
    # nsfw_path = os.listdir("../../dAIsy21App/lfr_api/media/media/anihotime/shit/")
    # nsfw_path = np.array(nsfw_path)
    # nsfw_path=pd.DataFrame(nsfw_path,columns=['Deck2 cards']) 
    # data = data.join(nsfw_path)
    price_values = np.random.uniform(0.015,0.04,[6000,1])
    price_df = pd.DataFrame(price_values,columns=['rand. prices for AI']) 
    data = data.join(price_df)
    data = data.join(bid_df) 
    data = data.join(stake_df)
    data = data.to_csv('c_force_data.csv',index=False)
    data = pd.read_csv('c_force_data.csv')
    # data = data.drop(['Deck1_cards'],axis=1) 
    # data = data.drop(['Deck2_cards'],axis=1)
    # data= data.drop(['Date'], axis =1)
    # data = data.drop('Adj Close',axis=1)
    X= data.drop(['Deck1_damage'],axis=1)

    y= data['Deck1_damage']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[-5:]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('c_force_data.csv')
    date = bata['rand. prices for AI']
    date = date.tail()
    print(date)
    bata = pd.read_csv('c_force_data.csv')
    date = bata['rand. prices for AI']
    print('PREDICTED price')
    y = c_force_model.predict(future_x)
    print(y[-1:])
    output =y[-1:]
    date = datetime.date.today()
    return render_template('c_force.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED PRICE FOR DATA IS {}'.format(output))


@app.route('/predict_litecoin',methods=['POST'])
def predict_litecoin():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ltcusd.csv')
    # data= data.drop(['Date'], axis =1)
    # data = data.drop('Adj Close',axis=1)
    X= data.drop(['close'],axis=1)
    y= data['close']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[-5:]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ltcusd.csv')
    date = bata['time']
    date = date.tail()
    print(date)
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ltcusd.csv')
    date = bata['time']
    print('PREDICTED Close')
    y = lit_model.predict(future_x)
    print(y[-1:])
    output =y[-1:]
    date = datetime.date.today()
    return render_template('crypto.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED Close FOR litecoin ON THE DAY OF {} IS $ {}'.format(date,output))


@app.route('/predict_xlm',methods=['POST'])
def predict_xlm():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_xlmusd.csv')
    # data= data.drop(['Date'], axis =1)
    # data = data.drop('Adj Close',axis=1)
    X= data.drop(['close'],axis=1)
    y= data['close']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[-5:]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_xlmusd.csv')
    date = bata['time']
    date = date.tail()
    print(date)
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_xlmusd.csv')
    date = bata['time']
    print('PREDICTED Close')
    y = xlm_model.predict(future_x)
    print(y[-1:])
    output =y[-1:]
    date = datetime.date.today()
    return render_template('crypto.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED Close FOR xlm  ON THE DAY OF {} IS $ {}'.format(date,output))




@app.route('/predict_bitcoin',methods=['POST'])
def predict_bitcoin():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
    # data= data.drop(['Date'], axis =1)
    # data = data.drop('Adj Close',axis=1)
    X= data.drop(['close'],axis=1)
    y= data['close']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[-5:]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
    date = bata['time']
    date = date.tail()
    print(date)
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
    date = bata['time']
    print('PREDICTED Close')
    y = bit_model.predict(future_x)
    print(y[-1:])
    output =y[-1:]
    date = datetime.date.today()
    return render_template('crypto.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED Close FOR Bitcoin ON THE DAY OF {} IS $ {}'.format(date,output))

@app.route('/predict_ethereum',methods=['POST'])
def predict_ethereum():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ethusd.csv')
    # data= data.drop(['Date'], axis =1)
    # data = data.drop('Adj Close',axis=1)
    X= data.drop(['close'],axis=1)
    y= data['close']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[-5:]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ethusd.csv')
    date = bata['time']
    date = date.tail()
    print(date)
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ethusd.csv')
    date = bata['time']
    print('PREDICTED Close')
    y = eth_model.predict(future_x)
    print(y[-1:])
    output =y[-1:]
    date = datetime.date.today()
    return render_template('crypto.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED Close FOR Ethereum ON THE DAY OF {} IS $ {}'.format(date,output))

@app.route('/predict_APPLE',methods=['POST'])
def predict_APPLE():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
    data = data.fillna(28.630752329973355)
    data= data.drop(['Date'], axis =1)
    data = data.drop('Adj Close',axis=1)
    X= data.drop(['Close'],axis=1)
    y= data['Close']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[-5:]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
    date = bata['Date']
    date = date.tail()
    print(date)
    bata = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
    date = bata['Date']
    print('PREDICTED Close')
    y = AAPL_model.predict(future_x)
    print(y[-1:])
    output =y[-1:]
    date = datetime.date.today()
    return render_template('stocks.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED Close FOR APPLE ON THE DAY OF {} IS $ {}'.format(date,output))

@app.route('/predict_MSFT',methods=['POST'])
def predict_MSFT():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
    data= data.drop(['Date'], axis =1)
    data = data.drop('Adj Close',axis=1)
    X= data.drop(['Close'],axis=1)
    y= data['Close']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[-5:]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
    date = bata['Date']
    date = date.tail()
    print(date)
    bata = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
    date = bata['Date']
    print('PREDICTED Close')
    y = MSFT_model.predict(future_x)
    print(y[-1:])
    output =y[-1:]
    date = datetime.date.today()
    return render_template('stocks.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED Close FOR MICROSOFT ON THE DAY OF {} IS $ {}'.format(date,output))


@app.route('/predict_ROKU',methods=['POST'])
def predict_ROKU():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
    data= data.drop(['Date'], axis =1)
    data = data.drop('Adj Close',axis=1)
    X= data.drop(['Close'],axis=1)
    y= data['Close']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    # X = X[-7:8696]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
    date = bata['Date']
    date = date.tail()
    print(date)
    bata = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
    date = bata['Date']
    print('PREDICTED Close')
    y = ROKU_model.predict(future_x)
    print(y[-1:])
    output =y[-1:]
    date = datetime.date.today()
    return render_template('stocks.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED Close FOR ROKU ON THE DAY OF {} IS $ {}'.format(date,output))
@app.route('/predict_api',methods=['GET'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
    data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
    data= data.drop(['Date'], axis =1)
    # data = data.drop('Adj Close',axis=1)
    X= data.drop(['Close'],axis=1)
    y= data['Close']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[-1:]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
    date = bata['Date']
    date = date.tail()
    print(date)
    bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
    date = bata['Date']
    print('PREDICTED Close')
    y = model.predict(future_x)
    print(y[-1:])

    output =y[-1:]

    return jsonify(output)
@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1",port=3000) #debug=True,host="0.0.0.0",port=50000
