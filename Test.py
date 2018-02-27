import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import neural_network
import matplotlib.pyplot as plt

train_percent = 0.80

companies = ['Reliance_Industries', 'Indian_Oil_Corporation', 'ONGC', 
             'Oil_India', 'Bharat_Petroleum', 'Hindustan_Petroleum']

oilDataset = pd.read_csv('oil_prices.csv')

oilDataset.columns = ['date', 'oil_price']

oilDataset['date'] = pd.to_datetime(oilDataset['date'])

removed_features = ['Open', 'High', 'Low', 'Volume', 'Close']

print("Pearson Coefficients :-")

for company in companies:
    stockDataset = pd.read_csv(company + '.csv')
    stockDataset.drop(removed_features, axis = 1, inplace=True)
    stockDataset.columns = ['date', 'adj_close']
    stockDataset['date'] = pd.to_datetime(stockDataset['date'])
    
    dataset = stockDataset.merge(oilDataset, on='date', how='right')
    dataset['oil_price'] = pd.to_numeric(dataset['oil_price'], errors='coerce')
    dataset['adj_close'] = pd.to_numeric(dataset['adj_close'], errors='coerce')
    dataset = dataset.dropna(0)
    
    datasetWithoutDate = dataset.drop('date', axis=1)
    datasetWithoutDate = datasetWithoutDate[datasetWithoutDate.columns].astype(float)    
    
    pearson, _ = pearsonr(dataset['oil_price'], dataset['adj_close'])
    
    print(company + " \t: " + str(pearson))
    
x_train, x_test, y_train, y_test = train_test_split(datasetWithoutDate['oil_price'],
                                                    datasetWithoutDate['adj_close'], 
                                                                      train_size=train_percent,
                                                                      test_size=1-train_percent)
    
regressor = RandomForestRegressor()
#regressor = linear_model.LinearRegression() 
#regressor = neural_network.MLPRegressor()


regressor.fit(pd.DataFrame(x_train), pd.DataFrame(y_train))

y_pred = regressor.predict(pd.DataFrame(x_test))

plt.figure()

plt.subplot(221)
plt.scatter(x=x_test, y=y_test)
plt.title('Actual')

plt.xlabel('Oil Price in Dollar per Unit')
plt.ylabel('Adjusted Closing Price')

plt.subplot(222)
plt.scatter(x=x_test, y=y_pred)
plt.title('Predicted')

plt.xlabel('Oil Price in Dollar per Unit')
plt.ylabel('Adjusted Closing Price')

plt.subplot(223)
plt.scatter(x=x_test, y=y_test)
plt.scatter(x=x_test, y=y_pred)
plt.legend(['Actual', 'Predicted'])
plt.title('Both')

plt.xlabel('Oil Price in Dollar per Unit')
plt.ylabel('Adjusted Closing Price')

plt.tight_layout()


