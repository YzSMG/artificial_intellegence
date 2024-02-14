import pandas as pd #importing packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(r"C:\Users\naifb\OneDrive\Documents\Python\artificial intellegence\datasets\delaney solubility with descriptors.cs") #adding variables
y = df['logS']
x = df.drop('logS', axis=1)
lr = LinearRegression()
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=100)

# training the model
lr.fit(x_train, y_train)
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
lr_train_me = mean_squared_error(y_train, y_lr_test_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# testing the model
lr_test_me = mean_squared_error(y_test, y_lr_test_pred) 
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print('lr me (Train) ', lr_train_me)
print('lr r2 (Train) ', lr_train_r2)
print('lr me (test) ', lr_test_me)
print('lr r2 (test) ', lr_test_r2)