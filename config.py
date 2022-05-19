import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


def generate_params(gbm=True, gbm_learning_rate=[0.01, 0.1], gbm_max_depth=[3, 8],
                     gbm_n_estimators=[500, 1000], gbm_subsample=[1, 0.5, 0.7]):

    if gbm:
        gbm_params = {"learning_rate": gbm_learning_rate,
                    "max_depth": gbm_max_depth,
                    "n_estimators": gbm_n_estimators,
                    "subsample": gbm_subsample}
    

    regressors = [("GBM", GradientBoostingRegressor(), gbm_params)]

    return regressors


def hyperparameter_optimization(X, y, cv=3,gbm=True, gbm_learning_rate=[0.01, 0.1], gbm_max_depth=[3, 8],
                     gbm_n_estimators=[500, 1000], gbm_subsample=[1, 0.5, 0.7]):
    regressors = generate_params(gbm, gbm_learning_rate, gbm_max_depth,
                     gbm_n_estimators, gbm_subsample)
    print("Hyperparameter Optimization....")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        final_model_before = regressor.fit(X_train, y_train)
        y_pred = final_model_before.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE BEFORE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_).fit(X_train, y_train)

        y_pred = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"RMSE AFTER: {round(rmse, 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    return final_model




