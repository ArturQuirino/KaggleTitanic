import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

train_file_path = 'train.csv'
titanic_data = pd.read_csv(train_file_path)

columns_of_interest = ['Pclass', 'SibSp', 'Parch']


X = titanic_data[columns_of_interest]
y = titanic_data.Survived

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

titanic_model = RandomForestRegressor()
titanic_model.fit(train_X, train_y)

val_predictions = titanic_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))