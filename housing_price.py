# imports
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


import pickle

from helper import get_cat_num_cols, get_missing_percentage, get_missing_columns_percentage, extractor, get_multicollinearity, drop_columns, drop_rows, ModelTraining

# loading data
test_set_path = 'data/test.csv'
train_set_path = 'data/train.csv'

test_set = pd.read_csv(test_set_path)
train_set = pd.read_csv(train_set_path)

total_column_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType','MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'MSSubClass', 'BsmtFinSF2', '3SsnPorch', 'PoolArea', 'MiscVal', 'EnclosedPorch', 'ScreenPorch', 'OverallCond', 'KitchenAbvGr', 'BedroomAbvGr']

train_set_cleaned = drop_columns(train_set, total_column_to_drop)
test_set_cleaned = drop_columns(test_set, total_column_to_drop)

# Drop rows
train_set_cleaned = drop_rows(train_set_cleaned)


# Extract features and target
X_train_val, y_train_val = extractor(train_set_cleaned, 'SalePrice')


# Split the training data to train and validation set
train_X, val_X, train_y, val_y = train_test_split(X_train_val, y_train_val, train_size=0.8, test_size=0.2, random_state=0)


# Training and Prediction

# Instantiate models
model_instance_dt = ModelTraining(train_X, train_y)
model_instance_rf = ModelTraining(train_X, train_y, 'rf')
model_instance_xgb = ModelTraining(train_X, train_y, 'xgb')


# Train models
train_model_1 = model_instance_dt.train()
train_model_2 = model_instance_rf.train()
train_model_3 = model_instance_xgb.train()


# Validate models

validate_models_1 = model_instance_dt.validate(val_X, val_y, train_model_1)
validate_models_2 = model_instance_rf.validate(val_X, val_y, train_model_2)
validate_models_3 = model_instance_xgb.validate(val_X, val_y, train_model_3)

print(validate_models_1)
print(validate_models_2)
print(validate_models_3)


# Perform prediction on test_set
predict = model_instance_xgb.predict(test_set_cleaned[:1], train_model_3)
print(predict['result'])

# save the models as a pickle file
# dt_model_pkl_file = "models/housing_price_model_dt.pkl"
# rf_model_pkl_file = "models/housing_price_model_rf.pkl"
# xgb_model_pkl_file = "models/housing_price_model_xgb.pkl"

# pickle.dump(train_model_1, open(dt_model_pkl_file, 'wb'))
# pickle.dump(train_model_2, open(rf_model_pkl_file, 'wb'))
# pickle.dump(train_model_3, open(xgb_model_pkl_file, 'wb'))


# # Load the models
# loaded_dt_model = pickle.load(open(dt_model_pkl_file, 'rb'))
# loaded_rf_model = pickle.load(open(rf_model_pkl_file, 'rb'))
# loaded_xgb_model = pickle.load(open(xgb_model_pkl_file, 'rb'))


# Perform prediction on loaded model
# print(loaded_xgb_model['model'].predict(test_set_cleaned[:1]))



