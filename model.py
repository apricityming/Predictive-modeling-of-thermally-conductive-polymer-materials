import xgboost as xgb
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold


def xgboost_grid_search(X_train, y_train, param_grid, cv=5):
    xgb_reg = XGBRegressor(objective="reg:squarederror", random_state=42)
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Negative MSE:", grid_search.best_score_)
    best_xgb_model = grid_search.best_estimator_

    return best_xgb_model

def randomforest_grid_search(X_train, y_train, param_grid, cv=5):
    rnd_reg = RandomForestRegressor(bootstrap=False, random_state=42)
    grid_search = GridSearchCV(estimator=rnd_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Negative MSE:", grid_search.best_score_)
    best_rnd_model = grid_search.best_estimator_

    return best_rnd_model

def SVR_grid_search(X_train, y_train, param_grid, cv=5):
    svr_reg = SVR(random_state=42)
    grid_search = GridSearchCV(estimator=svr_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Negative MSE:", grid_search.best_score_)
    best_svr_model = grid_search.best_estimator_

    return best_svr_model

def xgboost_predict(X_train, y_train, X_test, y_test, best_params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    xgb_reg = XGBRegressor(objective="reg:squarederror", random_state=42, **best_params)
    xgb_reg.fit(X_train, y_train)
    train_predict = xgb_reg.predict(X_train)
    test_predict = xgb_reg.predict(X_test)
    rmse_scores = -1 * cross_val_score(xgb_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    mae_scores = -1 * cross_val_score(xgb_reg, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
    r2_train_score = r2_score(train_predict, y_train)
    r2_test_score = r2_score(test_predict, y_test)
    print("r2 train:", r2_train_score)
    print("r2 test:", r2_test_score)
    print("RMSE Scores:", rmse_scores)
    print("MAE Scores:", mae_scores)
    average_rmse = rmse_scores.mean()
    average_mae = mae_scores.mean()
    print("Average RMSE across 5 folds:", average_rmse)
    print("Average MAE across 5 folds:", average_mae)

    return xgb_reg, train_predict, test_predict

def randomforest_predict(X_train, y_train, X_test, y_test, best_params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rnd_reg = RandomForestRegressor(bootstrap=False,, random_state=42, **best_params)
    rnf_reg.fit(X_train, y_train)
    train_predict = rnd_reg.predict(X_train)
    test_predict = rnd_reg.predict(X_test)
    rmse_scores = -1 * cross_val_score(rnd_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    mae_scores = -1 * cross_val_score(rnd_reg, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
    r2_train_score = r2_score(train_predict, y_train)
    r2_test_score = r2_score(test_predict, y_test)
    print("r2 train:", r2_train_score)
    print("r2 test:", r2_test_score)
    print("RMSE Scores:", rmse_scores)
    print("MAE Scores:", mae_scores)
    average_rmse = rmse_scores.mean()
    average_mae = mae_scores.mean()
    print("Average RMSE across 5 folds:", average_rmse)
    print("Average MAE across 5 folds:", average_mae)

    return rnd_reg, train_predict, test_predict

def SVR_preidct(X_train, y_train, X_test, y_test, best_params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    svr_reg = SVR(random_state=42, **best_params)
    svr_reg.fit(X_train, y_train)
    train_predict = svr_reg.predict(X_train)
    test_predict = svr_reg.predict(X_test)
    rmse_scores = -1 * cross_val_score(svr_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    mae_scores = -1 * cross_val_score(svr_reg, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
    r2_train_score = r2_score(train_predict, y_train)
    r2_test_score = r2_score(test_predict, y_test)
    print("r2 train:", r2_train_score)
    print("r2 test:", r2_test_score)
    print("RMSE Scores:", rmse_scores)
    print("MAE Scores:", mae_scores)
    average_rmse = rmse_scores.mean()
    average_mae = mae_scores.mean()
    print("Average RMSE across 5 folds:", average_rmse)
    print("Average MAE across 5 folds:", average_mae)

    return svr_reg, train_predict, test_predict