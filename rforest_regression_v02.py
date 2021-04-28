### Dependencies
import pandas as pd #data frames and other stuff
import matplotlib.pyplot as plt #basic graphs
import pickle

# Usefull functions
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence

# Used models
from sklearn.ensemble import RandomForestRegressor

#%reset -f

SEED = 500

# Load insurance_train_test.pickle file with  X_train, y_train, X_test, y_test variables
with open(r"insurance_train_test.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)

# Instantiate RandomForestRegressor 'rf' 5000 estimators
forest_model_0 = RandomForestRegressor(n_estimators=1000,
                                   min_samples_split=0.05,
                                   max_features = 4,
                                   random_state=SEED)


# Fit 'rf' to the training set
forest_model_0.fit(X_train, y_train.values.ravel()) #values.ravel() flattened array expected by RandomForestRegressor
# Predict the test set labels 'y_pred'
y_pred_0 = forest_model_0.predict(X_test)
# Evaluate the test set RMSE
rmse_test_0 = mean_squared_error(y_test, y_pred_0, squared=False)
print(rmse_test_0)

###########################################
###Exhaustive grid optimization using CV###
###########################################

# Create the  grid
hyper_grid = {'n_estimators': [3000,5000],
               'max_features': [4,8,11],
               'min_samples_split': [10, 20]}

#reinstantiate RandomForestRegressor regressor with empty parameter set
forest_model_cv = RandomForestRegressor()

# Instantiate the GridSearchCV with forest_model_cv  as estimator
grid_search = GridSearchCV(estimator = forest_model_cv, param_grid = hyper_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train.values.ravel())
#best parameters
print(grid_search.best_params_)
#best estimator
forest_model_opt= grid_search.best_estimator_ #best_estimator_
# Predict the test set labels 'y_pred'
y_pred_cv = forest_model_opt.predict(X_test)
rmse_test_cv = mean_squared_error(y_test, y_pred_cv, squared=False)
print(rmse_test_cv)
print(rmse_test_0)

##########################
####variable importance###
##########################
# Create a pd.Series of features impurity based importances (np array)
#feature_importances_
importances_rf = pd.Series(forest_model_opt.feature_importances_, index = X_train.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen') 
plt.show()

###Permutation Based Feature Importance###
#use permutation_importance()
perm_importance_rf = permutation_importance(forest_model_opt, X_test, y_test).importances_mean
perm_importance_rf = pd.Series(perm_importance_rf, index = X_train.columns) #convert to pd series with index = feature names

# Sort perm_importances_rf
perm_importance_rf = perm_importance_rf.sort_values() #sort_values()
# Make a horizontal bar plot
perm_importance_rf.plot(kind='barh', color='lightgreen'); plt.show()

###############################
####partial dependence plots###
###############################
#plot_partial_dependence(estimator, X, features)
plot_partial_dependence(forest_model_opt, X_train, ["bmi", "age"])
#twodimensional (tuple) ("bmi", "age")
plot_partial_dependence(forest_model_opt, X_train, [("bmi", "age")] )

#pdp for smoker_yes==0
#filter only smoker_yes==0
X_train_tmp = X_train[X_train['smoker_yes']==1]
#plot pdp only for filtered values in X_train_tmp
plot_partial_dependence(forest_model_opt, X_train_tmp, ["bmi", "age"] )
