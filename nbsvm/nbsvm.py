import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

class NBSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    NBSVM Classifier: Currently supports binary classification
    
    Attributes:
        C (float): Penalty parameter C of the error term.
        beta (float): between 0 and 1; level of interpolation between MNB and NBSVM
        alpha (float): smoothing factor 
        probability (bool): Whether to enable probability estimates
        class_weight: Set the parameter C of class i to class_weight[i]*C for SVC.
                      If not given, all classes are supposed to have weight one.
                      The “balanced” mode uses the values of y to automatically adjust weights
                      inversely proportional to class frequencies in the input data as
                      n_samples / (n_classes * np.bincount(y))
    """
    
    def __init__(self, C=1.0, beta=0.5, alpha=1, probability=False, class_weight='balanced'):
        self.C = C
        self.beta = beta
        self.alpha = alpha
        self.r = None
        self.coef_ = None
        self.intercept_ = None
        self.interpolated_coef_ = None
        self.interpolated_intercept_ = None
        self.probability = probability
        self.class_weight = class_weight
        self.platt_model_ = None
        
    def _log_probs(self, X, y):
        p_sum = np.sum(X[y==1], axis=0) + self.alpha
        n_sum = np.sum(X[y==0], axis=0) + self.alpha
        p_tot = len(X[y==1]) + self.alpha
        n_tot = len(X[y==0]) + self.alpha
        p_ratio = p_sum/p_tot
        n_ratio = n_sum/n_tot
        self.r = np.log(p_ratio/n_ratio)
        return self
       
    def _binarize(self, X):
        X[X>0] = 1
        return X
    
    def _interpolate(self, weights):
        return (self.beta*weights) + (1-self.beta)*(np.mean(abs(weights)))  
    
    def fit(self, X, y):
        "Fit the NBSVM model according to the given training data."
        X_binarized = X.copy()
        X_binarized = self._binarize(X_binarized)
        self._log_probs(X_binarized, y)
        del X_binarized
        X_nb = X*self.r
        svm = SVC(C=self.C, kernel='linear', class_weight=self.class_weight)
        svm.fit(X_nb, y)
        self.coef_ = svm.coef_
        self.intercept_ = svm.intercept_
        self.interpolated_coef_ = self._interpolate(svm.coef_)
        self.interpolated_intercept_ = self._interpolate(svm.intercept_)
        self.model = svm
        if self.probability:
            print(y.sum(), type(y))
            self._platt_scale(X, y)
        return self
        
    def predict(self, X):
        "Perform classification on samples in X."
        X_nb = X*self.r
        preds = np.add(np.dot(X_nb, self.interpolated_coef_.transpose()), self.interpolated_intercept_)
        preds[preds>0] = 1
        preds[preds<=0] = 0 
        return preds
    
    def decision_function(self, X):
        "Evaluates the decision function for the samples in X."
        X_nb = X*self.r
        dec_fn = np.add(np.dot(X_nb, self.interpolated_coef_.transpose()),  self.interpolated_intercept_)
        return dec_fn
    
    def fit_transform(self, X, y):
        """
        Fit the NBSVM model according to the given training data and
        return classification results on the training data
        """
        self.fit(X, y)
        preds = self.predict(X)
        return preds
    
    def predict_proba(self, X):
        "Perform classification on samples in X and return class-wise probability scores"
        if not self.probability:
            raise ValueError("probabilty was set to False")
        else:
            dec_fn = self.decision_function(X)
            return self.platt_model_.predict(dec_fn)
        
    def _platt_scale(self, X, y):
        # NOT WORKING WELL
        n_tot = len(y[y==0])
        p_tot = len(y[y==1])
        n_tar = 1/(n_tot + 2)
        p_tar = (p_tot + 1)/(p_tot + 2)
        y[y==1] = p_tar
        y[y==0] = n_tar
        dec_fn = self.decision_function(X)
        logit = LogitRegression()
        logit.fit(dec_fn, y)
        self.platt_model_ = logit
        return self
    
    def score(self, X, y):
        "Returns the mean accuracy on the given test data and labels."
        preds = self.predict(X)
        return 100*round(np.mean(y==preds), 2)
      
from sklearn.linear_model import LinearRegression

#helper class
class LogitRegression(LinearRegression):
    
    def fit(self, X, p):
        p = np.asarray(p)
        y = np.log(p / (1 - p))
        return super().fit(X, y)

    def predict(self, X):
        y = super().predict(X)
        return 1 / (np.exp(-y) + 1)
