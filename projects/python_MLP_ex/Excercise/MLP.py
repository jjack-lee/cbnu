# %%

from sklearn.neural_network import MLPRegressor
import pandas as pd

car_df = pd.read_csv('./ToyotaCorolla_preprocessed.csv')
car_df

# car_df.set_index('Fuel_Type_Petrol', drop=True, inplace=True)
# car_df

X = ['Age', 'Milage_KM']
y = 'Price'

data_X = car_df[X]
data_y = car_df[y]

mlp = MLPRegressor(hidden_layer_sizes=15, max_iter=1000)
mlp.fit(data_X, data_y)

newData = pd.DataFrame(
    {'Age': 0.1, 'Milage_KM': 0.1}, index=[1])
newData

mlp.predict(newData)

# %%
