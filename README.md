### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
1. model.py - This contains code for our Machine Learning model to perform stock predictions based on the past training data in 'BTC-USD.csv', etc... files from yahoo finance for quick prototyping / DATA RESTRUCTING EACH APP UPDATE TO RANDOMIZE PRICES

2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.


### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py, eth_model, etc...
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 3000.

3. Navigate to URL http://localhost:3000 or 127.0.0.1/3000

Choose the dApp/ App

AFTERWARDS, THE  PREDICTION WILL BE DISPLAYED AFTER PAYMENT WHEN LAUNCHED TO MAINNET!
"# c-force-reg" 
