"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import numbers
import warnings
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from sklearn.base import MetaEstimatorMixin, BaseEstimator, ClassifierMixin, clone, is_classifier, is_regressor
from sklearn.utils import check_random_state, Bunch
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.calibration import CalibratedClassifierCV



import pandas as pd
import matplotlib.pyplot as plt
from random import random
from math import exp, sqrt, log, tanh
from copy import deepcopy
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# class baseSafePredict(BaseEstimator):
#     def __init__(self, )
# def _set_random_states(estimator, random_state=None):
#     random_state = check_random_state(random_state)
#     to_set={}
#     for key in sorted(estimator.get_params(deep=True)):
#         if key== 'random_state' or key.endswith('__random_state'):
#             to_set[key] = random_state.randint(np.iinfo(np.int32).max)
#     if to_set:
#         estimator.set_params(**to_set)
# class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
#     """
#     Parameters
#     ----------
#     base_estimator: object, optional, default=None
#     n_estimators: integer
#     estimator_params: list of strings

#     Attributes
#     ----------
#     base_estimator_ : estimator
#     estimators_ : list of estimators
#     """
#     _required_parameters=[]
#     @abstractmethod
#     def __init__(self, base_estimator, n_estimators=2, estimator_params=tuple()):
#         self.base_estimator = base_estimator
#         self.n_estimators = n_estimators
#         self.estimator_params = estimator_params

#     def _validate_estimator(self, default=None):
#         if not isinstance(self.n_estimators, numbers.Integral):
#             raise ValueError("n_estimators must be an integer, "
#                              " got {0}".format(type(self.n_estimators)))
#         if self.n_estimators <= 0:
#             raise ValueError("n_estimators must be greater than zero "
#                              "got {0}".format(self.n_estimators))
#         if self.base_estimator is not None:
#             self.base_estimator_ = self.base_estimator
#         else:
#             self.base_estimator_ = default
        
#         if self.base_estimator_ is None:
#             raise ValueError("base estimator cannot be None")

#     def _make_estimator(self, append=True, random_state = None):
#         estimator = clone(self.base_estimator_)
#         estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
#         if random_state is not None:
#             _set_random_states(estimator, random_state)
#         if append:
#             self.estimators_.append(estimator)
#         return estimator
#     def __len__(self):
#         return len(self.estimators_)
#     def __getitem__(self, index):
#         return self.estimators_[index]
#     def __iter__(self):
#         return iter(self.estimators_)

# class _BaseHeterogenousEnsemble(MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta):
    
#     _required_parameters = ['estimators']

#     @property
#     def named_estimators(self):
#         return Bunch(**dict(self.estimators))

#     @abstractmethod
#     def __init__(self, estimators):
#         self.estimators = estimators

#     def _validate_estimators(self):
#         if self.estimators is None or len(self.estimators) == 0:
#             raise ValueError(
#                 "Invalid 'estimators' attribute, 'estimators' should be a list"
#                 " of (string, estimator) tuples."
#             )
#         names, estimators = zip(*self.estimators)
#         # defined by MetaEstimatorMixin
#         self._validate_names(names)

        # FIXME: deprecate the usage of None to drop an estimator from the
#         # ensemble. Remove in 0.24
#         if any(est is None for est in estimators):
#             warnings.warn(
#                 "Using 'None' to drop an estimator from the ensemble is "
#                 "deprecated in 0.22 and support will be dropped in 0.24. "
#                 "Use the string 'drop' instead.", FutureWarning
#             )

#         has_estimator = any(est not in (None, 'drop') for est in estimators)
#         if not has_estimator:
#             raise ValueError(
#                 "All estimators are dropped. At least one is required "
#                 "to be an estimator."
#             )

#         is_estimator_type = (is_classifier if is_classifier(self)
#                              else is_regressor)

#         for est in estimators:
#             if est not in (None, 'drop') and not is_estimator_type(est):
#                 raise ValueError(
#                     "The estimator {} should be a {}.".format(
#                         est.__class__.__name__, is_estimator_type.__name__[3:]
#                     )
#                 )

#         return names, estimators
        
#     def set_params(self, **params):
#         super()._set_params('estimators', **params)
#         return self
#     def get_params(self, deep=True):
#         return super()._get_params('estimators', deep=deep)
# def _parallel_build_estimators(n_estimators, ensemble, X, y, seeds, total_n_estimators, verbose):
#     estimators = []

#     for i in range(n_estimators):
#         if verbose > 1:
#             print("Building estimator %d of %d"
#                   "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

#         random_state = np.random.RandomState(seeds[i])
#         estimator = ensemble._make_estimator(append=False,
#                                              random_state=random_state)
#         estimator.fit(X, y)
#         estimators.append(estimator)
#     return estimators

# def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
#     n_samples = X.shape[0]
#     proba = np.zeros((n_samples, n_classes))
#     for estimator in zip(estimators):
#         if hasattr(estimator, "predict_proba"):
#             proba_estim = estimator.predict_proba(X)
#             if n_classes == len(estimator.classes_):
#                 proba += proba_estim
#             else:
#                 proba[:, estimator.classes_] += \
#                     proba_estim[:, range(len(estimator.classes_))]
#         else:
#             raise ValueError("predict_proba not supported by base estimator! ")


# class baseSafePredict(BaseEnsemble):
# class safePredict(MetaEstimatorMixin, BaseEstimator):
#     # @abstractmethod
#     def __init__(self, base_estimator=None, target_error = 0.1, dummy_prob = 0.5, prediction_prob = 0.5, alpha = 1, beta = 0, horizon = 1, refusal_class = -1, random_state=None):
#         self.target_error = target_error
#         self.dummy_prob = dummy_prob
#         self.prediction_prob = prediction_prob
#         self.alpha = alpha
#         self.beta = beta
#         self.horizon = horizon
#         self.refusal_class = refusal_class
#         self.random_state = random_state
    
#     def fit(self, X, y):
#         return self._fit(X, y, random_state=self.random_state)

#     def _fit(self, X, y, random_state = None):
#         # for i in range(len(X)):
#         self._partial_fit(X, y, random_state=random_state)
#         return self

#     def _first_call_partial_fit(self):

#         super()._validate_estimator()
#         if getattr(self, 'k_', None) is None:
            
#             self.estimator_ = super()._make_estimator(append=False)
#             self.k_ = 1
#             self.time_ = 0
#             self.loss_ = []
#             self.y_preds_ = []
#         # if getattr(self, 'learning_rate_', None) is None:
#             # self.learning_rate_ = np.sqrt((-np.log(self.dummy_prob) - (self.horizon-1)*np.log(self.alpha))/((1-self.target_error)*(2**(self.k_/2))))
#             # if getattr(self, 'classifier_', None) is None:
#             #     self.classifier_ = deepcopy(self.classifier)
#             # self._initializeSafePredict()
        
#         if getattr(self, 'init_dummy_prob_', None) is None and self.time_ == 0:
#             self.dummy_prob_ = deepcopy(self.dummy_prob)
#             self.prediction_prob_ = deepcopy(self.prediction_prob)
#             self.prediction_prob_return_ = 0
#             self.init_dummy_prob_ = deepcopy(self.dummy_prob)
#             self.init_prediction_prob_ = deepcopy(self.prediction_prob)
#             self.probs_ = [deepcopy(self.init_dummy_prob_),deepcopy(self.init_prediction_prob_)]
#             self.variance_sum_ = deepcopy(self.init_prediction_prob_*self.init_dummy_prob_)
#             self.learning_rate_ = np.sqrt((-np.log(self.init_dummy_prob_) - (self.horizon-1)*np.log(self.alpha))/((1-self.target_error)*(2**(self.k_/2))))
#         return

#     def _partial_fit(self, X, y, random_state = None):
#         X,y = check_X_y(X,y)
#         random_state = check_random_state(random_state)
#         self._first_call_partial_fit()
        
#         if getattr(self.estimator_, "partial_fit", None) is None or not callable(getattr(self.estimator_, "partial_fit", None)):
#             self.estimator_.fit(X,y)
#         else:
#             self.estimator_.partial_fit(X,y)
         
#         for i in range(len(X)):
#             y_pred = self._predict(X[i], random_state=random_state)
#             self.loss_.append(y_pred == y[i])
#             if self.variance_sum_ < 2**self.k_:
#                 self.dummy_prob_ = self.dummy_prob_*np.exp(-self.learning_rate_*self.target_error)
#                 self.prediction_prob_ = self.prediction_prob_*np.exp(-self.learning_rate_*self.loss_[self.time_])  #Here X is assuming it is a loss sequence. 
#                 sum_probabilities = self.dummy_prob_ + self.prediction_prob_
#                 self.dummy_prob_ /= sum_probabilities[0]
#                 self.prediction_prob_ /= sum_probabilities[0]

#             else:
#                 self.dummy_prob_ = self.init_dummy_prob_
#                 self.prediction_prob_ = self.init_prediction_prob_
#                 self.k_ +=1
#                 self.learning_rate_ = self.learning_rate_/np.sqrt(2)
#                 self.variance_sum_ = self.dummy_prob_*self.prediction_prob_

#             self.prediction_prob_return_ = self.prediction_prob_

#             self.dummy_prob_ = self.dummy_prob_*(self.beta-self.alpha) + 1 - self.beta
#             self.prediction_prob_ = self.prediction_prob_*(self.beta-self.alpha) + self.alpha

#             pred_prob = np.asarray(self.prediction_prob_)
#             if pred_prob.ndim >= 2:
#                 self.variance_sum_ += self.prediction_prob_[0]*self.dummy_prob_
#             else:
#                 self.variance_sum_ += self.prediction_prob_*self.dummy_prob_
            

#             self.time_ += 1
#         return self
#     def partial_fit(self, X, y):
#         return self._partial_fit(X,y,self.random_state)
    
#     def predict(self, X):
#         return self._predict(X, self.random_state)
#     def _predict(self, X, random_state = None):
#         random_state = check_random_state(random_state)
#         predictions = []
#         check_shape_array_ = np.asarray(X, dtype=float)
#         if(check_shape_array_.ndim >= 2):
#             for x in X:
#                 x_array_ = np.asarray(x)
#                 y_pred = self.estimator_.predict(x_array_.reshape(1,-1))
#                 if (random_state.rand()>=self.prediction_prob_return_):
#                     y_pred = self.refusal_class
#                 predictions.append(y_pred)
#                 self.y_preds_.append(y_pred)
#         else:
#             y_pred = self.estimator_.predict(check_shape_array_.reshape(1,-1))
#             if (random_state.rand()>=self.prediction_prob_return_):
#                 y_pred = self.refusal_class
#             predictions.append(y_pred)
#             self.y_preds_.append(y_pred)
            
#         return predictions

class safePredict(MetaEstimatorMixin, BaseEstimator):
    def __init__(self, base_estimator=None, base_estimator_params = tuple(), target_error = 0.1, dummy_prob = 0.5, prediction_prob = 0.5, alpha = 0, beta = 1, horizon = 1, refusal_class = -1, calibration=None, random_state=None):
        self.base_estimator = base_estimator
        self.base_estimator_params = base_estimator_params
        self.target_error = target_error
        self.dummy_prob = dummy_prob
        self.prediction_prob = prediction_prob
        self.alpha = alpha
        self.beta = beta
        self.horizon = horizon
        self.refusal_class = refusal_class
        self.random_state = random_state
        self.calibration = calibration

    def _validate_estimator(self, default=None):
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default
        
        if self.base_estimator_ is None:
            raise ValueError("base estimator cannot be None")
        return
    
    
    def _set_random_states(self, estimator, random_state=None):
        random_state = check_random_state(random_state)
        to_set={}
        for key in sorted(estimator.get_params(deep=True)):
            if key== 'random_state' or key.endswith('__random_state'):
                to_set[key] = random_state.randint(np.iinfo(np.int32).max)
        if to_set:
            estimator.set_params(**to_set)
        return

    def fit(self, X, y):
        return self._fit(X, y, random_state=self.random_state)

    # def _predict_proba(self, X, y=None, random_state=None):
    #     if getattr(self.estimator_, "predict_proba", None) is None:
    def _calibrate(self, X, y, epsilon):
        if getattr(self.estimator_, "predict_proba", None) is None or not callable(getattr(self.estimator_, "predict_proba", None)):
            self.estimator_ = self._make_estimator(random_state=self.random_state, calibration_method="sigmoid")
        probabilities = self.estimator_.predict_proba(X)
        predictions = self.estimator_.predict(X)
        num_rows, num_columns = X.shape
        potential_thresholds = probabilities.max(axis=1)
        smallest_thresholds = np.argsort(potential_thresholds)
        errors = sum(predictions != y)
        num_preds = num_rows
        threshold = 1

        if errors/num_preds > epsilon:
            for i in smallest_thresholds:
                if predictions[i] != y[i]:
                    errors = errors-1
                num_preds = num_preds - 1
                if errors/num_preds <= epsilon:
                    threshold = potential_thresholds[i]
                    break
        else:
            threshold = 0
        return threshold


    def _cv_calibrate(self, X, y, epsilon, k=3):
        kf = KFold(n_splits=k, shuffle=False)
        thresholds = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.estimator_.fit(X_train, y_train)
            thresholds.append(self._calibrate(X_test, y_test, epsilon))
        self.estimator_.fit(X,y)
        return np.mean(thresholds)

    def _conf_test(self, X_test, threshold):
        test_probs = self.estimator_.predict_proba(X_test)
        predictions = self.estimator_.predict(X_test)
        refused = np.sum(test_probs>threshold, axis=1) == 0
        return predictions, refused

    def _fit(self, X, y, random_state = None):
        # for i in range(len(X)):
        if getattr(self, "estimator_", None) is None:
            self._validate_estimator(RandomForestClassifier())
            self.estimator_ = self._make_estimator(random_state)
            self._set_random_states(self.estimator_, random_state)
        self._partial_fit(X, y, random_state=random_state)
        # self.refusal_class_ = y[0]
        # self.estimator_.fit(X,y)
        return self
    def _make_estimator(self, random_state = None,calibration_method= None):
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.base_estimator_params})
        if self.calibration is not None:
            if self.calibration == "sigmoid" or self.calibration == "isotonic":
                estimator = CalibratedClassifierCV(estimator, method=self.calibration, cv=3)
            else:
                raise ValueError("Calibration type must be 'isotonic' or 'sigmoid'.")
        elif calibration_method is not None:
            if calibration_method == "sigmoid" or calibration_method == "isotonic":
                estimator = CalibratedClassifierCV(estimator, method=calibration_method, cv=3)
            else:
                raise ValueError("Calibration type must be 'isotonic' or 'sigmoid'.")
        if random_state is not None:
            self._set_random_states(estimator, random_state)
        


        return estimator

    def _first_call_partial_fit(self):

        self._validate_estimator(RandomForestClassifier())
        if getattr(self, 'k_', None) is None:
            # self._validate_estimator(default=RandomForestClassifier())

            self.estimator_ = self._make_estimator(random_state=self.random_state)
            self._set_random_states(self.estimator_, random_state=self.random_state)
            self.k_ = 1
            self.time_ = 0
            self.loss_ = []
            self.y_preds_ = []
            self.refusal_class_ = deepcopy(self.refusal_class)
        # if getattr(self, 'learning_rate_', None) is None:
            # self.learning_rate_ = np.sqrt((-np.log(self.dummy_prob) - (self.horizon-1)*np.log(self.alpha))/((1-self.target_error)*(2**(self.k_/2))))
            # if getattr(self, 'classifier_', None) is None:
            #     self.classifier_ = deepcopy(self.classifier)
            # self._initializeSafePredict()
        
        if getattr(self, 'init_dummy_prob_', None) is None and self.time_ == 0:

            self.dummy_prob_ = deepcopy(self.dummy_prob)

            self.prediction_prob_ = deepcopy(1-self.dummy_prob)

            self.prediction_prob_return_ = 0
            self.init_dummy_prob_ = deepcopy(self.dummy_prob)
            self.init_prediction_prob_ = deepcopy(1-self.dummy_prob)
            self.probs_ = [deepcopy(self.init_dummy_prob_),deepcopy(self.init_prediction_prob_)]
            self.variance_sum_ = deepcopy(self.init_prediction_prob_*self.init_dummy_prob_)
            self.learning_rate_ = np.sqrt(-np.log(self.init_dummy_prob_) - (self.horizon-1)*np.log(1-self.alpha))/((1-self.target_error)*(2**(self.k_/2)))
        return

    def _partial_fit(self, X, y, random_state = None):
        X,y = check_X_y(X,y)
        random_state = check_random_state(random_state)
        self._first_call_partial_fit()
        
        if getattr(self.estimator_, "partial_fit", None) is None or not callable(getattr(self.estimator_, "partial_fit", None)):
            self.estimator_.fit(X,y)
        else:
            self.estimator_.partial_fit(X,y)
        for i in tqdm(range(len(X))):
            self._predict(X[i], y[i], random_state)
        # self._update(X,y, random_state)
        return self
    def _update(self, X, y, random_state = None):
        y_pred = self._predict(np.reshape(X, (1,-1)), random_state=random_state)
        # print("Len loss:", len(self.loss_))
        # print("time:", (self.time_))
        if y_pred[0]!= -1:
            if y_pred[0] == y:
                self.loss_.append(0)
            else:
                self.loss_.append(1)
        else:
            self.loss_.append(0)
        # self.loss_.append(y_pred == y)
        if self.variance_sum_ < 2**self.k_:

            self.dummy_prob_ = self.dummy_prob_*np.exp((-self.learning_rate_)*self.target_error)
            self.prediction_prob_ = self.prediction_prob_*np.exp((-self.learning_rate_)*self.loss_[self.time_])  
            sum_probabilities = self.dummy_prob_ + self.prediction_prob_
            self.dummy_prob_ /= sum_probabilities
            self.prediction_prob_ /= sum_probabilities


        else:

            self.dummy_prob_ = self.init_dummy_prob_
            self.prediction_prob_ = self.init_prediction_prob_
            self.k_ +=1
            self.learning_rate_ = self.learning_rate_/np.sqrt(2)
            self.variance_sum_ = self.init_dummy_prob_*self.init_prediction_prob_

        self.prediction_prob_return_ = self.prediction_prob_

        self.dummy_prob_ = self.dummy_prob_*(self.beta-self.alpha) + 1 - self.beta
        self.prediction_prob_ = self.prediction_prob_*(self.beta-self.alpha) + self.alpha

      
        self.variance_sum_ += self.prediction_prob_*self.dummy_prob_
        

        self.time_ += 1
        
        return self
    def partial_fit(self, X, y):
        return self._partial_fit(X,y,self.random_state)
    
    def predict(self, X, y=None):
        return self._predict(X, y, self.random_state)
    def _predict(self, X, y=None, random_state = None):
        random_state = check_random_state(random_state)
        # predictions = self.estimator_.predict(X)
        predictions = []
        # check_shape_array_ = np.asarray(X, dtype=float)
        # if(check_shape_array_.ndim >= 2):
        do_update = False
        if y is not None:
            do_update = True
        if not isinstance(y, list):
            x_array_ = np.asarray(X)
            y_pred = self.estimator_.predict(x_array_.reshape(1,-1))[0]
            if (random_state.rand()>=self.prediction_prob_return_):
                # print("Pred prob:",self.prediction_prob_return_)
                # y_pred = np.array([self.refusal_class])
                y_pred = self.refusal_class
            predictions.append(y_pred)
            # print(y_pred)
            self.y_preds_.append(y_pred)
            if(do_update):
                self._update(X, y, random_state)
        else:
            for i in range(len(X)):
                x_array_ = np.asarray(X[i])
                y_pred = self.estimator_.predict(x_array_.reshape(1,-1))[0]
                if (random_state.rand()>=self.prediction_prob_return_):
                    # print("Pred prob2:", self.prediction_prob_return_)
                    # y_pred = np.array([self.refusal_class])
                    y_pred = self.refusal_class
                predictions.append(y_pred)
                # print(y_pred)
                self.y_preds_.append(y_pred)
                if(do_update):
                    self._update(X[i], y[i], random_state)
        # else:
        #     y_pred = self.estimator_.predict(check_shape_array_.reshape(1,-1))
        #     if (random_state.rand()>=self.prediction_prob_return_):
        #         y_pred = self.refusal_class
        #     predictions.append(y_pred)
        #     self.y_preds_.append(y_pred)
        pred_result = np.asarray(predictions, dtype=np.float32)
        return pred_result


# test = safePredict()
# print(test.random_state)
# test = TemplateClassifier(classifier=RandomForestClassifier(n_estimators=100, min_samples_leaf=1, oob_score=False, n_jobs=-1))
# check_estimator(test)
# check_estimator(RandomForestClassifier(n_estimators=100, min_samples_leaf=1, oob_score=False, n_jobs=-1))


# X=[]
# y = []
# print("getting data")
# X, y = fetch_openml('codrna', version=1, return_X_y=True)
# y = y.astype(float)
# y = (y + 1)/2
# y = y.astype(int);

# holdout_X, X, holdout_y, y = train_test_split(X, y, train_size=2500/len(y))
# scaler = StandardScaler(with_mean=False)
# holdout_X = scaler.fit_transform(holdout_X)
# X = scaler.transform(X)
# X, rest_X, y, rest_y = train_test_split(X, y, train_size=10000/len(y))
# T = y.size
# numCP = 1
# if (numCP != 0):
#     for i in range(1, numCP + 1):
#         permuted_labels = np.random.permutation(np.arange(min(y[((i)*(T//(numCP+1))):((i+1)*(T//(numCP+1)))]),max(y[((i)*(T//(numCP+1))):((i+1)*(T//(numCP+1)))])+1))
#         for t in range((i)*(T//(numCP+1)),(i+1)*(T//(numCP+1))):
#             y[t] = permuted_labels[y[t]] 
# test_sp = safePredict(target_error=0.05, horizon=holdout_X.shape[0])
# print("Here")
# test_sp.fit(holdout_X.toarray(), holdout_y)
# print("Here2")
# epsilon = 1-test_sp.estimator_.score(rest_X, rest_y)
# print("Epsilon:", epsilon)
# tau = 100
# tl = tau
# horizon = X.shape[0]
# sp =safePredict(target_error=epsilon,alpha=(1/horizon), horizon=X.shape[0])
# predictions = []
# sp.fit(X[:tl].toarray(),y[:tl])
# for t in tqdm(range(tl, horizon)):
#     if t % tau == 0:
#         sp.estimator_.fit(X[:t], y[:t])
#     predictions.append(sp.predict(X[t].toarray(), y[t])[0])
# print("SP Preds:",predictions[5000:5100])
# # print(len(sp.loss_))
# # predictions= sp.predict(holdout_X.toarray(), holdout_y)
# errors = 0
# counter_sp = 0
# error_rates = []
# total_preds = len(predictions)
# print("Total predictions sp: ", total_preds)
# for i in tqdm(range(total_preds)):
#     if predictions[i] != (-1):
#         if predictions[i] != y[i+tl]:
#             errors = errors+1
#     counter_sp = counter_sp + 1
#     if counter_sp == 0:
#         error_rates.append(0)
#     else:
#         error_rates.append(errors/counter_sp)

# print("SP Error rate: ", errors/total_preds)
# model = RandomForestClassifier()
# model.fit(X[:tl].toarray(), y[:tl])

# base_predictions = []
# for t in tqdm(range(tl, horizon)):
#     if t%tau == 0:
#         model.fit(X[:t].toarray(), y[:t])
#     base_predictions.append(model.predict(X[t].toarray())[0])

# base_err = 0
# counter = 0
# base_error_rates = []
# total_base_preds = len(base_predictions)
# for i in tqdm(range(total_base_preds)):
#     if base_predictions[i] != y[i+tl]:
#         base_err = base_err+1
#     counter = counter + 1
#     base_error_rates.append(base_err/counter)
# print("Base error rate: ", base_err/total_base_preds)
# threshold = []
# for i in range(total_preds):
#     threshold.append(0.1)
# plt.plot(np.linspace(0,total_preds, total_preds), error_rates)
# plt.plot(np.linspace(0,total_preds, total_preds), base_error_rates)
# plt.plot(np.linspace(0,total_preds, total_preds), threshold)
# plt.show()