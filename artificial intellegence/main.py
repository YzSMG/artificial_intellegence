import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/YzSMG/datasets/main/delaney%20solubility%20with%20descriptors.csv")
print(df)
# Separate features and target
y = df['logS']
x = df.drop('logS', axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=100)

# Train the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict the test data
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Calculate the test metrics
lr_train_me = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_me = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
#print("lr me (train) ", lr_train_me)
#print("lr r2 (train) ", lr_train_r2)
#print("lr me (test) ", lr_test_me)
#print("lr r2 (test) ", lr_test_r2)

# printing results
lr_results = pd.DataFrame(['linear regression', lr_train_me, lr_train_r2, lr_test_me, lr_train_r2]).transpose()
lr_results.columns = ("Method", "training me", "training_r2", "testing_me", "testing_r2")

# training the model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# applying the model to make a prediction
y_rf_train_pred = lr.predict(x_train)
y_rf_test_pred = lr.predict(x_test)

rf_train_me = mean_squared_error(y_train, y_lr_train_pred)
rf_train_r2 = r2_score(y_train, y_lr_train_pred)

rf_test_me = mean_squared_error(y_test, y_lr_test_pred)
rf_test_r2 = r2_score(y_test, y_lr_test_pred)

#printing results

rf_results = pd.DataFrame(['random forest', rf_train_me, rf_train_r2, rf_test_me, rf_test_r2]).transpose()
rf_results.columns = ("Method", "training me", "training_r2", "testing_me", "testing_r2")
print(rf_results)

# getting the best model
df_models = pd.concat([lr_results, rf_results], axis=0)
df_models.reset_index(drop=True)

import matplotlib.pyplot as plt
import numpy as np
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)

z = np.polyfit(y_train,y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), '#c71400')
plt.ylabel("predict LogS")
plt.xlabel("experimental LogS")
    