import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
df = pd.read_csv("ITSM_Dataset.csv")

print(df.head(3))
# print(df.columns)
# print(df['Product group'].unique())
# print(df['Source'].unique())
# print(df['Topic'].unique())
# print(df['Agent Group'].unique())
# print(df['Support Level'].unique())
# print(df['Priority'].unique())
# print(df['Country'].unique())

# encoding   data
ohe = OneHotEncoder(handle_unknown='ignore',
                    sparse_output=False,).set_output(transform='pandas')
ohe_transformer = ohe.fit_transform(df[['Product group',
                                        'Source',
                                        'Topic',
                                        'Agent Group',
                                        'Country'
                                        ]])
# mapping my ordinal  categories  to numerical
df['Support Level'] = df['Support Level'].map({'L1': 1, 'L2': 2, 'L3': 3})
df['Priority'] = df['Priority'].map(
    {'Low': 0,  'Medium': 1, 'High': 2, 'Critical': 3})
# concatenating  the new column  and dropping the previous one
df = pd.concat([df.reset_index(drop=True), ohe_transformer.reset_index(drop=True)], axis=1).drop(columns=['Product group',
                                                                                                          'Source',
                                                                                                          'Topic',
                                                                                                          'Agent Group',
                                                                                                          'Country'])
# converting the created time and resolution times into proper datatimeformat
df['Created time'] = pd.to_datetime(df['Created time'])
df['Resolution time'] = pd.to_datetime(df['Resolution time'])
# Crearing resolution hours from created and resolution time which will serve as the target variable
df['Resolution Hours'] = (
    df['Resolution time']-df['Created time']).dt.total_seconds()/3600
# Dropping columns that could cause leakage  or affect the model
df = df.drop(columns=['Status',
                      'Ticket ID',
                      'Agent Name',
                      'Created time',
                      'Expected SLA to resolve',
                      'Expected SLA to first response',
                      'First response time',
                      'SLA For first response',
                      'Resolution time',
                      'SLA For Resolution',
                      'Close time',
                      'Agent interactions', 'Survey results',
                      'Latitude', 'Longitude'])
# splitting  the dataset into features (X) and target (y) variable
X = df.drop(columns='Resolution Hours')
y = df['Resolution Hours']
# Training and testing the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=34)

rd = RandomForestRegressor(random_state=12)
rd.fit(X_train, y_train)
# predicting the model
y_pred = rd.predict(X_test)
mea = mean_absolute_error(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
r2_s = r2_score(y_pred, y_test)
# tunning my dataset for a better result
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rd_cv = GridSearchCV(estimator=rd, param_grid=param_grid,
                     cv=2,
                     scoring='neg_mean_squared_error',
                     n_jobs=-1)


rd_cv.fit(X_train, y_train)
y_pred2 = rd_cv.predict(X_test)
mea = mean_absolute_error(y_pred2, y_test)
mse = mean_squared_error(y_pred2, y_test)
r2_s = r2_score(y_pred2, y_test)
print(mea)
print(mse)
print(r2_s)
print(rd_cv.best_params_)

y_test_reset = y_test.reset_index(drop=True)
# creating a Dataframe to compare predictions with actual values
results = pd.DataFrame({
    "Actual Resolution Hours": y_test_reset,
    "predicted Resolution Hours": y_pred2
})
# making my results a csv file
results.to_csv("resolution_prediction.csv", index=False)
print(results.head())
