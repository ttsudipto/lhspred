from .resource import DataResource, storeResourceCSV
from .density import execute as densityExec, compute_positiveness, compute_negativeness, split_pos_neg
from sklearn.model_selection import train_test_split
from .svr import gridSearchRBF, gridSearchPoly, gridSearchLinear
from .mlpr import gridSearch1Layer, gridSearch2Layer
from .pickler import saveModel, testModel, load_model_from_file
from .model import Model
import numpy as np

res1 = DataResource('derivation.csv')
res1.read()
print(res1.data.shape, res1.target.shape)

res2 = DataResource('validation.csv')
res2.read()
print(res2.data.shape, res2.target.shape)

mergedData = np.concatenate((res1.data, res2.data), axis=0)
mergedTarget = np.concatenate((res1.target, res2.target), axis=0)
print(mergedData.shape, mergedTarget.shape)

XTrain, XTest, yTrain, yTest = train_test_split(mergedData, mergedTarget, test_size=0.1, random_state=1)
dres = DataResource('derivation.csv')
dres.data = XTrain
dres.target = yTrain
vres = DataResource('derivation.csv')
vres.data = XTest
vres.target = yTest
print(dres.data.shape, dres.target.shape)
print(vres.data.shape, vres.target.shape)

#from sklearn.model_selection import ShuffleSplit
#rs = ShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
#for train_index, test_index in rs.split(mergedData, mergedTarget):
    #i=0
#storeResourceCSV('derivation.csv', 'validation.csv', 'tt.csv', 'val.csv', test_index)
#print(sorted(train_index), sorted(test_index))

#CTScores1, outcomes1 = DataResource('derivation.csv').readProgressionData()
#print(CTScores1.shape, outcomes1.shape)
#CTScores2, outcomes2 = DataResource('validation.csv').readProgressionData()
#print(CTScores2.shape, outcomes2.shape)
#CTScores = np.concatenate((CTScores1, CTScores2), axis=None)
#outcomes = np.concatenate((outcomes1, outcomes2), axis=None)
#print(CTScores.shape, outcomes.shape)
#outcomesTrain, outcomesTest, yTr, yTe = train_test_split(outcomes, mergedTarget, test_size=0.1, random_state=1)
#CTScoresTrain, CTScoresTest, yTr, yTe = train_test_split(CTScores, mergedTarget, test_size=0.1, random_state=1)
#print(outcomesTrain.shape, outcomesTest.shape)
#print(CTScoresTrain.shape, CTScoresTest.shape)

########################################################
##                         F-test                     ##
########################################################
#from sklearn.feature_selection import f_regression, f_classif
#from scipy.stats import iqr
#from statistics import median
#medians = []
#iqrs = []
#maxs = []
#mins = []
#for j in range(dres.data.shape[1]) :
    #maxs.append(max(dres.data[:,j]))
    #mins.append(min(dres.data[:,j]))
    #iqrs.append(iqr(dres.data[:,j]))
    #medians.append(median(dres.data[:,j]))
#fval, pval = f_regression(dres.data, dres.target)
#print('Clinical features', 'Min', 'Max', 'IQR', 'Median', 'F-value', 'p-value')
#for i in range(len(fval)) :
    #print(dres.data_columns[i], round(mins[i], 3), round(maxs[i], 3), round(iqrs[i], 3), round(medians[i], 3), round(fval[i], 3), pval[i])

#posIndices, negIndices = split_pos_neg(outcomes)
#negCTscores = CTScores[negIndices]
#posCTscores = CTScores[posIndices]
#print(round(min(negCTscores), 3), round(max(negCTscores), 3), round(iqr(negCTscores), 3), round(median(negCTscores), 3))
#print(round(min(posCTscores), 3), round(max(posCTscores), 3), round(iqr(posCTscores), 3), round(median(posCTscores), 3))
#print(f_classif(CTScores.reshape(-1,1), outcomes))

########################################################
##                        Density                     ##
########################################################
#densityExec(CTScores, outcomes, 1) # all
#densityExec(CTScores, outcomes, 2) # negatives
#densityExec(CTScores, outcomes, 3) # positives

#import numpy as np
#scores = np.linspace(0, 25, 100)
#for score in scores :
    #print(score, round(compute_negativeness(score), 4)*100, round(compute_positiveness(score), 4)*100)

#scores = [(0, 2), (2.1, 4), (4.1, 6), (6.1, 8), (8.1, 10), (10.1, 12), 
          #(12.1, 14), (14.1, 16), (16.1, 18), (18.1, 20), (20.1, 22), (22.1, 25)]
#for sp in scores :
    #print(int(sp[0]), int(sp[1]), 
          #round(compute_negativeness(sp[0])*100, 2), round(compute_negativeness(sp[1])*100, 2),
          #round(compute_positiveness(sp[0])*100, 2), round(compute_positiveness(sp[1])*100, 2))


########################################################
##                 DecisionTreeRegressor              ##
########################################################
#from sklearn.tree import DecisionTreeRegressor as DTR
#dtrParams = {
        #'criterion' : 'mse',
        #'random_state' : 42
    #}
#dtr = DTR(**dtrParams)
#dtr.fit(dres.data, dres.target)

##yPred = dtr.predict(dres.data)
##score = dtr.score(dres.data, dres.target)
##for i in range(dres.target.shape[0]) :
    ##print(dres.target[i], yPred[i])
##print(score)

#yPred = dtr.predict(vres.data)
#score = dtr.score(vres.data, vres.target)
#for i in range(vres.target.shape[0]) :
    #print(vres.target[i], yPred[i])
#print(score)

########################################################
##                   LinearRegression                 ##
########################################################
#from sklearn.linear_model import LinearRegression
#linrParams = {
        #'normalize' : True
    #}
#linr = LinearRegression(**linrParams)
#linr.fit(dres.data, dres.target)

##yPred = linr.predict(dres.data)
##score = linr.score(dres.data, dres.target)
##for i in range(dres.target.shape[0]) :
    ##print(dres.target[i], yPred[i])
##print(score)

#yPred = linr.predict(vres.data)
#score = linr.score(vres.data, vres.target)
#for i in range(vres.target.shape[0]) :
    #print(vres.target[i], yPred[i])
#print(score)

########################################################
##                         SVR                        ##
########################################################
#gridSearchRBF(dres.data, dres.target, k=5)
#gridSearchPoly(dres.data, dres.target, k=5)
#gridSearchLinear(dres.data, dres.target, k=5)

svrParamsK3 = {
        'kernel' : 'rbf',
        'degree' : 3,
        'gamma' : 0.01,
        'coef0' : 1,
        'C' : 20,
        'epsilon' : 0.5
    }
svrParamsK5 = {
        'kernel' : 'rbf',
        'degree' : 3,
        'gamma' : 0.01,
        'coef0' : 1,
        'C' : 10,
        'epsilon' : 0.001
    }
svrParamsK10 = {
        'kernel' : 'rbf',
        'degree' : 3,
        'gamma' : 0.01,
        'coef0' : 1,
        'C' : 15,
        'epsilon' : 0.001
    }
svrParams = svrParamsK5
svr_k = 5

#model = Model('SVR', dres.data, dres.target, k=svr_k)
#model.learn(svrParams, scale=True)
#mseCV, maeCV, r2sCV, pccCV = model.predict_k_fold(scale=True)
#mseBCV, maeBCV, r2sBCV, pccBCV = model.predict_blind_data(vres.data, vres.target, scale=True)
#mseB, maeB, r2sB, pccB = model.predict_blind_without_CV(vres.data, vres.target, scale=True)
#print(svrParams['kernel'], mseCV, maeCV, r2sCV, pccCV, mseBCV, maeBCV, r2sBCV, pccBCV, mseB, maeB, r2sB, pccB)
#acc, sens, spec = model.predict_risk_class_k_fold(outcomesTrain, scale=True)
#print(acc, sens, spec)
#acc, sens, spec = model.predict_risk_class_blind_CV(vres.data, outcomesTest, scale=True)
#print(acc, sens, spec)
#acc, sens, spec = model.predict_risk_class_blind_without_CV(vres.data, outcomesTest, scale=True)
#print(acc, sens, spec)

#testModel('SVR', svrParams, dres.data, dres.target, vres.data, vres.target)
##saveModel('SVR', svrParams, dres.data, dres.target, storeScalers=True)
#estimator = load_model_from_file('SVR')
#dataScaler, targetScaler = load_model_from_file('Scalers')
#model = Model('SVR', dres.data, dres.target, k=5)
#model.total_estimator = estimator
#mseB, maeB, r2sB, pccB = model.predict_blind_without_CV(vres.data, vres.target, scale=True)
#print(mseB, maeB, r2sB, pccB)
#print(model.get_params())

########################################################
##                        MLPR                        ##
########################################################
#gridSearch1Layer(dres.data, dres.target, k=5)
#gridSearch2Layer(dres.data, dres.target, k=5)

mlprParams = {
        'activation' : 'relu',
        'hidden_layer_sizes' : (200,),
        'learning_rate' : 'constant',
        'learning_rate_init' : 0.0001,
        'solver' : 'adam',
        'max_iter' : 10000,
        'alpha' : 0.0001,
        'random_state' : 1
    }
mlpr_k = 5

#model = Model('MLPR', dres.data, dres.target, k=mlpr_k)
#model.learn(mlprParams, scale=True)
#mseCV, maeCV, r2sCV, pccCV = model.predict_k_fold(scale=True)
#mseBCV, maeBCV, r2sBCV, pccBCV = model.predict_blind_data(vres.data, vres.target, scale=True)
#mseB, maeB, r2sB, pccB = model.predict_blind_without_CV(vres.data, vres.target, scale=True)
#print(mlprParams['hidden_layer_sizes'], mseCV, maeCV, r2sCV, pccCV, mseBCV, maeBCV, r2sBCV, pccBCV, mseB, maeB, r2sB, pccB)
#acc, sens, spec = model.predict_risk_class_k_fold(outcomesTrain, scale=True)
#print(acc, sens, spec)
#acc, sens, spec = model.predict_risk_class_blind_CV(vres.data, outcomesTest, scale=True)
#print(acc, sens, spec)
#acc, sens, spec = model.predict_risk_class_blind_without_CV(vres.data, outcomesTest, scale=True)
#print(acc, sens, spec)

#testModel('MLPR', mlprParams, dres.data, dres.target, vres.data, vres.target)
##saveModel('MLPR', mlprParams, dres.data, dres.target, storeScalers=True)
#estimator = load_model_from_file('MLPR')
#dataScaler, targetScaler = load_model_from_file('Scalers')
#model = Model('MLPR', dres.data, dres.target, k=5)
#model.total_estimator = estimator
#mseB, maeB, r2sB, pccB = model.predict_blind_without_CV(vres.data, vres.target, scale=True)
#print(mseB, maeB, r2sB, pccB)
#print(model.get_params())
