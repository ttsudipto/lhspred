from .resource import DataResource
from .svr import predict as sPredict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLPR
import numpy as np

dres = DataResource('derivation.csv')
dres.read()
print(dres.data.shape, dres.target.shape)

vres = DataResource('validation.csv')
vres.read()
print(vres.data.shape, vres.target.shape)

mergedData = np.concatenate((dres.data, vres.data), axis=0)
mergerTarget = np.concatenate((dres.target, vres.target), axis=0)
print(mergedData.shape, mergerTarget.shape)

XTrain, XTest, yTrain, yTest = train_test_split(mergedData, mergerTarget, test_size=0.3, random_state=42)
trainres = DataResource('derivation.csv')
trainres.data = XTrain
trainres.target = yTrain
testres = DataResource('derivation.csv')
testres.data = XTest
testres.target = yTest

#print(min(yTrain), max(yTrain), yTrain)
#print(min(yTest), max(yTest), yTest)

svrParamGridRBF = {
        'kernel' : ['rbf'],
        'gamma' : [1e-2, 1e-3, 1e-4],
        'C' : [5, 10, 15, 20],
        'epsilon' : [0.01, 0.1, 0.5, 0.8]
    }
svrParamGridPoly = {
        'kernel' : ['poly'],
        'degree' : [2, 3],
        'gamma' : [1e-2, 1e-3, 1e-4],
        'coef0' : [1, 2, 3],
        'C' : [1, 5, 10, 15],
        'epsilon' : [0.01, 0.1, 0.5, 0.8]
    }
svrParamGridLinear = {
        'kernel' : ['linear'],
        'C' : [1, 5, 10, 15],
        'epsilon' : [0.01, 0.1, 0.5, 0.7]
    }

def gridSearchRBF(trainResource, testResource) :
    print('kernel', 'C', 'gamma', 'epsilon', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    dataScaler = StandardScaler()
    targetScaler = StandardScaler()
    for c in svrParamGridRBF['C'] :
        for g in svrParamGridRBF['gamma'] :
            for e in svrParamGridRBF['epsilon'] :
                svr = SVR(kernel='rbf', C=c, gamma=g, epsilon=e)
                #svr.fit(trainResource.data, trainResource.target)
                svr.fit(dataScaler.fit_transform(trainResource.data), targetScaler.fit_transform(trainResource.target.reshape(-1,1)).ravel())
                mse, mae, r2s, pcc, yPred = sPredict(svr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
                print('RBF', c, g, e, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

def gridSearchPoly(trainResource, testResource) :
    print('kernel', 'degree', 'C', 'gamma', 'coef0', 'epsilon', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    dataScaler = StandardScaler()
    targetScaler = StandardScaler()
    for d in svrParamGridPoly['degree'] :
        for c in svrParamGridPoly['C'] :
            for g in svrParamGridPoly['gamma'] :
                for co in svrParamGridPoly['coef0'] :
                    for e in svrParamGridPoly['epsilon'] :
                        svr = SVR(kernel='poly', degree=d, C=c, gamma=g, coef0=co, epsilon=e)
                        #svr.fit(trainResource.data, trainResource.target)
                        svr.fit(dataScaler.fit_transform(trainResource.data), targetScaler.fit_transform(trainResource.target.reshape(-1,1)).ravel())
                        mse, mae, r2s, pcc, yPred = sPredict(svr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
                        print('poly', d, c, g, co, e, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

def gridSearchLinear(trainResource, testResource) :
    print('kernel', 'C', 'epsilon', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    dataScaler = StandardScaler()
    targetScaler = StandardScaler()
    for c in svrParamGridLinear['C'] :
        for e in svrParamGridLinear['epsilon'] :
            svr = SVR(kernel='linear', C=c, epsilon=e)
            #svr.fit(trainResource.data, trainResource.target)
            svr.fit(dataScaler.fit_transform(trainResource.data), targetScaler.fit_transform(trainResource.target.reshape(-1,1)).ravel())
            mse, mae, r2s, pcc, yPred = sPredict(svr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
            print('linear', c, e, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

#gridSearchRBF(trainres, testres)
#gridSearchPoly(trainres, testres)
#gridSearchLinear(trainres, testres)

mlprParamGrid1Layer = {
        'activation' : ['relu', 'logistic'],
        'hidden_layer_sizes' : [(50,), (100,), (150,), (200,), (250,), (300,)],
        'learning_rate' : ['constant'],
        'learning_rate_init' : [0.001, 0.0001, 0.00001],
        'alpha' : [0.0001],
        #'momentum' : [0.4, 0.7, 0.9]
    }

mlprParamGrid2Layer = {
        'activation' : ['relu', 'logistic'],
        'hidden_layer_sizes' : [(50,10), (100,20), (150,30), (200,40)],
        'learning_rate' : ['constant'],
        'learning_rate_init' : [0.001, 0.0001, 0.00001],
        'alpha' : [0.0001],
        #'momentum' : [0.4, 0.7, 0.9]
    }

def gridSearch1Layer(trainResource, testResource) :
    print('Activation', 'Learning_strategy', 'Layers', 'Learning_rate', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    #mlprParamGrid1Layer = mlprParamGrid1
    dataScaler = StandardScaler()
    targetScaler = StandardScaler()
    for a in mlprParamGrid1Layer['activation'] :
        for lr in mlprParamGrid1Layer['learning_rate'] :
            for hls in mlprParamGrid1Layer['hidden_layer_sizes'] :
                for lri in mlprParamGrid1Layer['learning_rate_init'] :
                    mlpr = MLPR(activation=a, solver='adam', random_state=42, max_iter=10000, learning_rate=lr, hidden_layer_sizes=hls, learning_rate_init=lri, alpha=0.0001)
                    #mlpr.fit(trainResource.data, trainResource.target)
                    #mse, mae, r2s, yPred = predict(mlpr, testResource.data, testResource.target, verbose=False)
                    #print(lr, hls, lri, a, round(mse, 3), round(mae, 3), round(r2s, 3))
                    mlpr.fit(dataScaler.fit_transform(trainResource.data), targetScaler.fit_transform(trainResource.target.reshape(-1,1)).ravel())
                    mse, mae, r2s, pcc, yPred = sPredict(mlpr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
                    print(a, lr, hls, lri, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

def gridSearch2Layer(trainResource, testResource) :
    print('Activation', 'Learning_strategy', 'Layers', 'Learning_rate', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    #mlprParamGrid2Layer = mlprParamGrid2
    dataScaler = StandardScaler()
    targetScaler = StandardScaler()
    for a in mlprParamGrid2Layer['activation'] :
        for lr in mlprParamGrid2Layer['learning_rate'] :
            for hls in mlprParamGrid2Layer['hidden_layer_sizes'] :
                for lri in mlprParamGrid2Layer['learning_rate_init'] :
                    mlpr = MLPR(activation=a, solver='adam', random_state=42, max_iter=10000, learning_rate=lr, hidden_layer_sizes=hls, learning_rate_init=lri, alpha=0.0001)
                    #mlpr.fit(trainResource.data, trainResource.target)
                    #mse, mae, r2s, yPred = predict(mlpr, testResource.data, testResource.target, verbose=False)
                    #print(lr, hls, lri, a, round(mse, 3), round(mae, 3), round(r2s, 3))
                    mlpr.fit(dataScaler.fit_transform(trainResource.data), targetScaler.fit_transform(trainResource.target.reshape(-1,1)).ravel())
                    mse, mae, r2s, pcc, yPred = sPredict(mlpr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
                    print(a, lr, hls, lri, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

#gridSearch1Layer(trainres, testres)
#gridSearch2Layer(trainres, testres)
