import numpy as np  # linear algebra

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import matplotlib.patches as mpatches

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# Regressors
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
import itertools
import matplotlib.pyplot as plt


def train_models(X, y, kfolds, models):
    kfold = KFold(kfolds)
    for model in models:
     # model['regressor'].fit(X,y)
        cf_result = cross_val_score(
            model['regressor'], X, y, cv=kfold, scoring='neg_mean_squared_log_error')
        model['cv_results'] = np.sqrt(cf_result*-1)
        msg = f"Regressor: {model['name']},   rmsle:{cf_result.mean().round(11)} "
        print(msg)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    '''Simplest Stacking approach : Averaging base models
       We begin with this simple approach of averaging base models. We build a new class to extend scikit-learn with our model and also to laverage encapsulation and code reuse (inheritance)
       Averaged base models class'''

    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    '''Less simple Stacking : Adding a Meta-model
        1. Split the total training set into disjoint sets (here traind amd .holdout)
        2. Train several base modles on the first part (train)
        3. ThTest these base models on the second part (holdout)
        4. Use the predictions from 3) (called out-of-folds predictions) as the
        inputs, and the correct responses (target variable) as the outputs to
        train a higher level learner called meta-model.
    '''

    def __init__(self, base_models, meta_model, n_folds):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X)
                             for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def sssplit(X, y, nsplits):
    '''This functions it to lead with unbaleced data set, and 
    make new balanced splits into train and test .
    '''
    sss = StratifiedKFold(n_splits=nsplits, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):
        print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    return original_Xtrain, original_Xtest, original_ytrain, original_ytest


class Class_Fit(object):
    """
    Fit and tuning an algorithim for classification
    """
    def __init__(self, clf, params=None):
        if params:            
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def grid_search(self, parameters, Kfold):
        self.grid = GridSearchCV(estimator = self.clf, param_grid = parameters, cv = Kfold)
        
    def grid_fit(self, X, Y):
        self.grid.fit(X, Y)
        
    def grid_predict(self, X, Y):
        self.predictions = self.grid.predict(X)
        print("Precision: {:.2f} % ".format(100*accuracy_score(Y, self.predictions)))

def tts_split(X, y, size, splits):
    '''Split the data in Train and
     test using the Shuffle split'''

    rs = ShuffleSplit(n_splits=splits, test_size=size)

    rs.get_n_splits(X)

    for train_index, test_index in rs.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

