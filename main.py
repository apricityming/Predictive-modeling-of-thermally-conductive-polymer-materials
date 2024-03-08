# -*- coding: utf-8 -*-
"""
The program was developed by the School of Chemistry and Chemical Engineering at CentralSouth University.
The conclusions mentioned in the paper are based on results from Intel's chip training. 
It is worth mentioning that although we did not test it for other hardware, it does not have a significant impact on the training results of the model
"""
import split_train_test
import preprocess_and_discretize
import warnings

warnings.filterwarnings("ignore")

target_column = 'Heat conduction/W¡¤m-1¡¤K-1'

dataset_file = './ML_PATH/datasets/paperdata.csv'
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [0, 0.1, 0.5, 1, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.45],
    'max_depth': [3, 5, 7, 9]
}
def run_model():
    print("Importing database and dividing dataset")
    X_standardized, y_discretized = preprocess_and_discretize(dataset_file, target_column)
    X_train,X_test,y_train,y_test = split_train_test(X, y, test_size=0.3, random_state=42)
    xgboost_grid_search(X_train, y_train, param_grid, cv=5)
    #randomforest_grid_search()
    #SVR_grid_search()
    print("Model is training...")

    xgb_reg,train_predict, test_predict

if __name__ == "__main__":
    run_model()
