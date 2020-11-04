import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.multiclass import type_of_target


class SoloModel(BaseEstimator):
    

    def __init__(self, estimator):
        self.estimator = estimator
        self.trmnt_preds_ = None
        self.ctrl_preds_ = None
        self._type_of_target = None

    def fit(self, X, y, treatment, estimator_fit_params=None):
        
        check_consistent_length(X, y, treatment)
        treatment_values = np.unique(treatment)
        if len(treatment_values) != 2:
            raise ValueError("Expected only two unique values, got %s" % len(treatment_values))

        if isinstance(X, np.ndarray):
            X_mod = np.column_stack((X, treatment))
        elif isinstance(X, pd.core.frame.DataFrame):
            X_mod = X.assign(treatment=treatment)
        else:
            raise TypeError("Expected numpy.ndarray or pandas.DataFrame in training vector X, got %s" % type(X))

        self._type_of_target = type_of_target(y)

        if estimator_fit_params is None:
            estimator_fit_params = {}
        self.estimator.fit(X_mod, y, **estimator_fit_params)
        return self

    def predict(self, X):
        
        if isinstance(X, np.ndarray):
            X_mod_trmnt = np.column_stack((X, np.ones(X.shape[0])))
            X_mod_ctrl = np.column_stack((X, np.zeros(X.shape[0])))
        elif isinstance(X, pd.core.frame.DataFrame):
            X_mod_trmnt = X.assign(treatment=np.ones(X.shape[0]))
            X_mod_ctrl = X.assign(treatment=np.zeros(X.shape[0]))
        else:
            raise TypeError("Expected numpy.ndarray or pandas.DataFrame in training vector X, got %s" % type(X))

        if self._type_of_target == 'binary':
            self.trmnt_preds_ = self.estimator.predict_proba(X_mod_trmnt)[:, 1]
            self.ctrl_preds_ = self.estimator.predict_proba(X_mod_ctrl)[:, 1]
        else:
            self.trmnt_preds_ = self.estimator.predict(X_mod_trmnt)
            self.ctrl_preds_ = self.estimator.predict(X_mod_ctrl)

        uplift = self.trmnt_preds_ - self.ctrl_preds_
        return uplift


class ClassTransformation(BaseEstimator):
  
    def __init__(self, estimator):
        self.estimator = estimator
        self._type_of_target = None

    def fit(self, X, y, treatment, estimator_fit_params=None):
        

        check_consistent_length(X, y, treatment)
        self._type_of_target = type_of_target(y)

        if self._type_of_target != 'binary':
            raise ValueError("This approach is only suitable for binary classification problem")
        
        _, treatment_counts = np.unique(treatment, return_counts=True)
        if treatment_counts[0] != treatment_counts[1]:
            warnings.warn(
                "It is recommended to use this approach on treatment balanced data. Current sample size is unbalanced.",
                category=UserWarning,
                stacklevel=2
            )

        y_mod = (np.array(y) == np.array(treatment)).astype(int)

        if estimator_fit_params is None:
            estimator_fit_params = {}
        self.estimator.fit(X, y_mod, **estimator_fit_params)
        return self

    def predict(self, X):
        
        uplift = 2 * self.estimator.predict_proba(X)[:, 1] - 1
        return uplift


class TwoModels(BaseEstimator):
    
    def __init__(self, estimator_trmnt, estimator_ctrl, method='vanilla'):
        self.estimator_trmnt = estimator_trmnt
        self.estimator_ctrl = estimator_ctrl
        self.method = method
        self.trmnt_preds_ = None
        self.ctrl_preds_ = None
        self._type_of_target = None

        all_methods = ['vanilla', 'ddr_control', 'ddr_treatment']
        if method not in all_methods:
            raise ValueError("Two models approach supports only methods in %s, got"
                             " %s." % (all_methods, method))

        if estimator_trmnt is estimator_ctrl:
            raise ValueError('Control and Treatment estimators should be different objects.')

    def fit(self, X, y, treatment, estimator_trmnt_fit_params=None, estimator_ctrl_fit_params=None):
     
        check_consistent_length(X, y, treatment)
        self._type_of_target = type_of_target(y)

        X_ctrl, y_ctrl = X[treatment == 0], y[treatment == 0]
        X_trmnt, y_trmnt = X[treatment == 1], y[treatment == 1]

        if estimator_trmnt_fit_params is None:
            estimator_trmnt_fit_params = {}
        if estimator_ctrl_fit_params is None:
            estimator_ctrl_fit_params = {}

        if self.method == 'vanilla':
            self.estimator_ctrl.fit(
                X_ctrl, y_ctrl, **estimator_ctrl_fit_params
            )
            self.estimator_trmnt.fit(
                X_trmnt, y_trmnt, **estimator_trmnt_fit_params
            )

        if self.method == 'ddr_control':
            self.estimator_ctrl.fit(
                X_ctrl, y_ctrl, **estimator_ctrl_fit_params
            )
            if self._type_of_target == 'binary':
                ddr_control = self.estimator_ctrl.predict_proba(X_trmnt)[:, 1]
            else:
                ddr_control = self.estimator_ctrl.predict_(X_trmnt)

            if isinstance(X_trmnt, np.ndarray):
                X_trmnt_mod = np.column_stack((X_trmnt, ddr_control))
            elif isinstance(X_trmnt, pd.core.frame.DataFrame):
                X_trmnt_mod = X_trmnt.assign(ddr_control=ddr_control)
            else:
                raise TypeError("Expected numpy.ndarray or pandas.DataFrame, got %s" % type(X_trmnt))

            self.estimator_trmnt.fit(
                X_trmnt_mod, y_trmnt, **estimator_trmnt_fit_params
            )

        if self.method == 'ddr_treatment':
            self.estimator_trmnt.fit(
                X_trmnt, y_trmnt, **estimator_trmnt_fit_params
            )
            if self._type_of_target == 'binary':
                ddr_treatment = self.estimator_trmnt.predict_proba(X_ctrl)[:, 1]
            else:
                ddr_treatment = self.estimator_trmnt.predict(X_ctrl)[:, 1]

            if isinstance(X_ctrl, np.ndarray):
                X_ctrl_mod = np.column_stack((X_ctrl, ddr_treatment))
            elif isinstance(X_trmnt, pd.core.frame.DataFrame):
                X_ctrl_mod = X_ctrl.assign(ddr_treatment=ddr_treatment)
            else:
                raise TypeError("Expected numpy.ndarray or pandas.DataFrame, got %s" % type(X_ctrl))

            self.estimator_ctrl.fit(
                X_ctrl_mod, y_ctrl, **estimator_ctrl_fit_params
            )

        return self

    def predict(self, X):
     
        if self.method == 'ddr_control':
            if self._type_of_target == 'binary':
                self.ctrl_preds_ = self.estimator_ctrl.predict_proba(X)[:, 1]
            else:
                self.ctrl_preds_ = self.estimator_ctrl.predict(X)

            if isinstance(X, np.ndarray):
                X_mod = np.column_stack((X, self.ctrl_preds_))
            elif isinstance(X, pd.core.frame.DataFrame):
                X_mod = X.assign(ddr_control=self.ctrl_preds_)
            else:
                raise TypeError("Expected numpy.ndarray or pandas.DataFrame, got %s" % type(X_mod))
            self.trmnt_preds_ = self.estimator_trmnt.predict_proba(X_mod)[:, 1]

        elif self.method == 'ddr_treatment':
            if self._type_of_target == 'binary':
                self.trmnt_preds_ = self.estimator_trmnt.predict_proba(X)[:, 1]
            else:
                self.trmnt_preds_ = self.estimator_trmnt.predict_proba(X)[:, 1]

            if isinstance(X, np.ndarray):
                X_mod = np.column_stack((X, self.trmnt_preds_))
            elif isinstance(X, pd.core.frame.DataFrame):
                X_mod = X.assign(ddr_treatment=self.trmnt_preds_)
            else:
                raise TypeError("Expected numpy.ndarray or pandas.DataFrame, got %s" % type(X_mod))
            self.ctrl_preds_ = self.estimator_ctrl.predict_proba(X_mod)[:, 1]

        else:
            if self._type_of_target == 'binary':
                self.ctrl_preds_ = self.estimator_ctrl.predict_proba(X)[:, 1]
                self.trmnt_preds_ = self.estimator_trmnt.predict_proba(X)[:, 1]
            else:
                self.ctrl_preds_ = self.estimator_ctrl.predict(X)
                self.trmnt_preds_ = self.estimator_trmnt.predict(X)

        uplift = self.trmnt_preds_ - self.ctrl_preds_

        return uplift
