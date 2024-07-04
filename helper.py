# Imports
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor


# function to get the categorical and numeric columns
def get_cat_num_cols(dataset):
  # get numeric and categorical columns
  numeric_cols = [col for col in dataset.columns if dataset[col].dtype != 'object']
  categorical_cols = [col for col in dataset.columns if dataset[col].dtype == 'object']
  return numeric_cols, categorical_cols

def get_missing_percentage(col_names, full_dataset, threshold=0):
    dataset = full_dataset[col_names]
    missing_percentage = {}
    for col in dataset.columns:
        val = round((dataset[col].isnull().sum()/dataset.shape[0])*100,2)
        if val > threshold:
            missing_percentage[col] = val
    return missing_percentage
    
# function to get columns missing value percentage
def get_missing_columns_percentage(dataset, type='full', threshold=0):
    num, cat = get_cat_num_cols(dataset)
    num_missing_percentage = get_missing_percentage(num, dataset, threshold)
    cat_missing_percentage = get_missing_percentage(cat, dataset, threshold)
    missing= []
    if type=='num':
        missing.append(num_missing_percentage)
        return missing
    elif type=='cat':
        missing.append(cat_missing_percentage)
        return missing
    else:
        return num_missing_percentage, cat_missing_percentage
  

# function to extract feature and label
def extractor(dataset, label):
  # drop rows with missing targets
  dataset.dropna(axis=0, subset=['SalePrice'], inplace=True)
  X = dataset.drop(label, axis=1)
  y = dataset[label]
  return X, y


# function to get collinearity
def get_multicollinearity(dataset, target_column=None, column1=None, column2=None):
    if column1 and column2:
        corr_mat = dataset[column1, column2].corr()
    elif target_column:
        corr_mat = None
        columns = dataset.columns
        for column in columns:
            if column.dtype != 'object':
                corr_mat = dataset[column, target_column].corr()
    else:
        corr_mat = dataset.corr(numeric_only=True)
    return corr_mat
    

# function to drop null-like columns
def drop_columns(dataset, columns):
    new_dataset = dataset.drop(columns=columns)
    return new_dataset


def drop_rows(dataset, threshold=5):
    subset_dic = get_missing_columns_percentage(dataset, threshold=threshold)
    subset_list = []
    for i in subset_dic:
        subset_list = list(i.keys())
    new_dataset = dataset.dropna(axis=0, subset=subset_list)
    return new_dataset


class ModelTraining:
    def __init__(self, X, y, method=None):
        self.method = method
        self.X = X
        self.y = y
        
        if method == 'rf':
            self.model = RandomForestRegressor(random_state=0)
            self.name = 'Random Forest'
        elif method == 'xgb':
            self.model = XGBRegressor(random_state=0)
            self.name = 'XGBoost'
        else:
            self.model = DecisionTreeRegressor(random_state=0)
            self.name = 'Decision Tree'
            self.parameters={
                "max_depth" : range(1,10),
                # "min_samples_leaf": range(1,5),
                # "min_samples_split": range(2,5),
            }

    def data_preprocessing(self, dataset):
        num, cat = get_cat_num_cols(dataset)
        numerical_transformer = Pipeline(steps=[
            ('num_imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler()),
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, num),
                ('cat', categorical_transformer, cat)
            ]
        )

        return preprocessor
        
    def train(self): 
        model_dic = {}
        processed_data = self.data_preprocessing(self.X)
        model_pipe = Pipeline(
            steps=[
                # ('preprocessor', processed_data),
                ('preprocessor', processed_data),
                ('model', self.model)
                ]
            )
        model_fit = model_pipe.fit(self.X, self.y) 
        model_dic['model'] = model_fit
        model_dic['name'] = self.name
        return model_dic
    
    def validate(self, val_X, val_y, model):
        predictions = {}
        prediction = model['model'].predict(val_X)
        mae = mean_absolute_error(val_y, prediction)

        predictions['name'] = model['name']
        predictions['score_type'] = "Mean Absolute Error"
        predictions['score'] = mae
        return predictions
    
    def predict(self, X, model):
        predictions = {}
        print(X.shape)
        prediction = model['model'].predict(X)
        predictions['X'] = X
        predictions['result'] = prediction
        return predictions
