# %%

from sklearn.linear_model import LinearRegression
import pandas as pd

car_df = pd.read_csv('./ToyotaCorolla_preprocessed.csv')
car_df

car_df.set_index('Fuel_Type_Petrol', drop=True, inplace=True)
car_df

X = ['Age', 'Milage_KM']
y = 'Price'

data_X = car_df[X]
data_y = car_df[y]

lm = LinearRegression()
lm.fit(data_X, data_y)

print('intercept (b0) ', lm.intercept_)
coef_names = ['b1', 'b2']
print(pd.DataFrame({'Predictor': data_X.columns,
                    'coefficient Name': coef_names,
                    'coefficient Value': lm.coef_}))

newData = pd.DataFrame(
    {'Age': 74, 'Milage_KM': 124057}, index=[1])
newData

lm.predict(newData)

# %%
