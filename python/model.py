from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, recall_score, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLPR
from statistics import mean
import numpy as np
import csv
import copy
from .density import compute_positiveness, compute_negativeness

class Model :
    """Class encapsulating an estimator class that implements a learning model of Scikit-learn and the data.

    It contains methods for training and testing the learning model with 
    k-fold cross validation. It also contains method to predict blind dataset.
    """
    
    def __init__(self, e_id, X, y, k=5, do_shuffle=False):
        """Constructor"""
        
        self.estimators = []
        self.total_estimator = None
        self.train_indices = []
        self.test_indices = []
        self.estimator_id = e_id
        if do_shuffle == True :
            self.data = shuffle(X, random_state=42)
            self.target = shuffle(y, random_state=42)
        else :
            self.data = X
            self.target = y
        self.n_folds = k
        self.split_CV_folds()
        self.dataScaler = StandardScaler().fit(self.data)
        self.targetScaler = StandardScaler().fit(self.target.reshape(-1,1))

    def create_estimator(self, params) :
        """Method that instantiates an estimator"""
        
        estimator = None
        if self.estimator_id == 'SVR' : ## SVM
            estimator = SVR()
        elif self.estimator_id == 'MLPR' : ## RF
            estimator = MLPR()
        estimator.set_params(**params)
        return estimator

    def split_CV_folds(self) :
        if self.n_folds == 1 :
            self.train_indices = [range(self.data.shape[0])]
            self.test_indices = [range(self.data.shape[0])]
        else :
            skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            for train_index, test_index in skf.split(self.data, self.target) :
                self.train_indices.append(train_index)
                self.test_indices.append(test_index)

    def learn_without_CV(self, params, scale=True) :
        estimator = self.create_estimator(params)
        if scale == True :
            estimator.fit(self.dataScaler.transform(self.data), self.targetScaler.transform(self.target.reshape(-1,1)).ravel())
        else :
            estimator.fit(self.data, self.target)
        self.total_estimator = copy.deepcopy(estimator)

    def learn_k_fold(self, params, scale=True) :
        self.estimators = []
        for f in range(self.n_folds) :
            estimator = self.create_estimator(params)
            if scale == True :
                estimator.fit(self.dataScaler.transform(self.data[self.train_indices[f]]), self.targetScaler.transform(self.target[self.train_indices[f]].reshape(-1,1)).ravel())
            else :
                estimator.fit(self.data[self.train_indices[f]], self.target[self.train_indices[f]])
            self.estimators.append(copy.deepcopy(estimator))
    
    def learn(self, params, scale=True) :
        self.learn_without_CV(params, scale)
        self.learn_k_fold(params, scale)
    
    def predict(self, estimator, X_test, scale=True) :
        if scale == True :
            y_pred = estimator.predict(self.dataScaler.transform(X_test))
            #print(y_pred.shape)
            y_pred = self.targetScaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        else :
            y_pred = estimator.predict(X_test)
        return y_pred
    
    def predict_k_fold(self, scale=True) :
        r2score = []
        mse = []
        mae = []
        pcc = []
        for f in range(self.n_folds) :
            y_test = self.target[self.test_indices[f]]
            y_pred = self.predict(self.estimators[f], self.data[self.test_indices[f]], scale)
            #y_pred = self.estimators[f].predict(self.data[self.test_indices[f]])
            r2score.append(r2_score(y_test, y_pred))
            mse.append(mean_squared_error(y_test, y_pred))
            mae.append(mean_absolute_error(y_test, y_pred))
            pcc.append(pearsonr(y_test, y_pred)[0])
            #if f == 0 :
                #for i in range(len(y_pred)):
                    #print(y_test[i], y_pred[i])
        return mean(mse), mean(mae), mean(r2score), mean(pcc)
    
    def predict_blind_data(self, b_data, b_target, scale=True) :
        """Method to perform prediction of blind dataset"""
        
        r2score = []
        mse = []
        mae = []
        pcc = []
        for f in range(self.n_folds) :
            y_pred = self.predict(self.estimators[f], b_data, scale)
            #y_pred = self.estimators[f].predict(b_data)
            r2score.append(r2_score(b_target, y_pred))
            mse.append(mean_squared_error(b_target, y_pred))
            mae.append(mean_absolute_error(b_target, y_pred))
            pcc.append(pearsonr(b_target, y_pred)[0])
        return mean(mse), mean(mae), mean(r2score), mean(pcc)

    def predict_blind_without_CV(self, b_data, b_target, scale=True) :
        y_pred = self.predict(self.total_estimator, b_data, scale)
        #y_pred = self.total_estimator.predict(b_data)
        return mean_squared_error(b_target, y_pred), mean_absolute_error(b_target, y_pred), r2_score(b_target, y_pred), pearsonr(b_target, y_pred)[0]
    
    def predict_risk_class_k_fold(self, risk_scores, scale=True) :
        sensitivities = []
        specificities = []
        accuracies = []
        for f in range(self.n_folds) :
            y_test = self.target[self.test_indices[f]]
            risk_test = risk_scores[self.test_indices[f]]
            y_pred = self.predict(self.estimators[f], self.data[self.test_indices[f]], scale)
            positiveness = [compute_positiveness(y) for y in y_pred]
            negativeness = [compute_negativeness(y) for y in y_pred]
            risk_pred = []
            for i in range(len(y_pred)) :
                if positiveness[i] > negativeness[i] :
                    risk_pred.append(1)
                else :
                    risk_pred.append(0)
            #for i in range(len(y_pred)) :
                #print(y_pred[i], y_test[i], positiveness[i], negativeness[i], risk_pred[i], risk_test[i])
            #print('')
            sensitivities.append(recall_score(risk_test, risk_pred))
            accuracies.append(accuracy_score(risk_test, risk_pred))
            tn, fp, fn, tp = confusion_matrix(risk_test, risk_pred).ravel()
            specificities.append((tn/1.0) / (tn+fp))
        return (mean(accuracies), mean(sensitivities), mean(specificities))

    def write_to_csv(self, filename, mae, mse, r2s) :
        """Method to write the results into a CSV file"""
        
        fields = ['MAE', 'MSE', 'R2-Score']
        csvfile = open(filename, 'w')
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i in range(len(mae)) :
            row = {fields[0]:mae[i], fields[1]:mse[i], fields[2]:r2s[i]}
            writer.writerow(row)
        csvfile.close()
