from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

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

#svrParamGrid = {
        #'kernel' : ['rbf'],
        #'degree' : [2, 3],
        #'gamma' : [1e-3],
        #'coef0' : [1],
        #'C' : [5],
        #'epsilon' : [0.1]
    #}

#def predict(svr, data, target, dataScaler=None, targetScaler=None, verbose=False) :
    #if dataScaler is None or targetScaler is None :
        #yPred = svr.predict(data)
        #mse = mean_squared_error(target, yPred, multioutput='uniform_average')
        #mae = mean_absolute_error(target, yPred, multioutput='uniform_average')
        #r2s = r2_score(target, yPred, multioutput='uniform_average')
        #pcc, pv = pearsonr(target, yPred)
    #else :
        #yPred = svr.predict(dataScaler.transform(data))
        #transformedTarget = targetScaler.transform(target.reshape(-1,1)).ravel()
        #mse = mean_squared_error(transformedTarget, yPred, multioutput='uniform_average')
        #mae = mean_absolute_error(transformedTarget, yPred, multioutput='uniform_average')
        #r2s = r2_score(transformedTarget, yPred, multioutput='uniform_average')
        #pcc, pv = pearsonr(transformedTarget, yPred)
        #yPred = targetScaler.inverse_transform(yPred)
    #if verbose == True :
        #print(str(mse), str(mae), str(r2s), str(pcc))
        #for i in range(target.shape[0]) :
            #print(target[i], yPred[i])
    
    #return (mse, mae, r2s, pcc, yPred)

def predict(model, data, target, dataScaler=None, targetScaler=None, verbose=False) :
    if dataScaler is None or targetScaler is None :
        yPred = model.predict(data)
        mse = mean_squared_error(target, yPred, multioutput='uniform_average')
        mae = mean_absolute_error(target, yPred, multioutput='uniform_average')
        r2s = r2_score(target, yPred, multioutput='uniform_average')
        pcc, pv = pearsonr(target, yPred)
    else :
        yPred = model.predict(dataScaler.transform(data))
        yPred = targetScaler.inverse_transform(yPred)
        transformedTarget = targetScaler.transform(target.reshape(-1,1)).ravel()
        mse = mean_squared_error(target, yPred, multioutput='uniform_average')
        mae = mean_absolute_error(target, yPred, multioutput='uniform_average')
        r2s = r2_score(target, yPred, multioutput='uniform_average')
        pcc, pv = pearsonr(target, yPred)
        #print(max(target), min(target), max(yPred), min(yPred))
    if verbose == True :
        print(str(mse), str(mae), str(r2s), str(pcc))
        for i in range(target.shape[0]) :
            print(target[i], yPred[i])
    
    return (mse, mae, r2s, pcc, yPred)

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
                mse, mae, r2s, pcc, yPred = predict(svr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
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
                        mse, mae, r2s, pcc, yPred = predict(svr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
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
            mse, mae, r2s, pcc, yPred = predict(svr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
            print('linear', c, e, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

#svr = SVR(**svrParams)
#svr.fit(dres.data, dres.target)

#svr.fit(StandardScaler().fit_transform(dres.data), StandardScaler().fit_transform(dres.target.reshape(-1,1)))
#vres.data = StandardScaler().fit_transform(vres.data)
#vres.target = StandardScaler().fit_transform(vres.target.reshape(-1,1))

#yPred = svr.predict(dres.data)
#score = svr.score(dres.data, dres.target)
#print(score)
#for i in range(dres.target.shape[0]) :
    #print(dres.target[i], yPred[i])

#predict(svr, data, target)
