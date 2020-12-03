from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

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

mlprParamGrid1 = {
        'activation' : ['relu'],
        'hidden_layer_sizes' : [(150,)],
        'learning_rate' : ['constant'],
        'learning_rate_init' : [0.0001],
        'alpha' : [0.0001],
        #'momentum' : [0.4, 0.7, 0.9]
    }
mlprParamGrid2 = {
        'activation' : ['relu'],
        'hidden_layer_sizes' : [(100,20)],
        'learning_rate' : ['constant'],
        'learning_rate_init' : [0.0001],
        'alpha' : [0.0001],
        #'momentum' : [0.4, 0.7, 0.9]
    }

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
                    mse, mae, r2s, pcc, yPred = predict(mlpr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
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
                    mse, mae, r2s, pcc, yPred = predict(mlpr, testResource.data, testResource.target, dataScaler=dataScaler, targetScaler=targetScaler, verbose=False)
                    print(a, lr, hls, lri, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))
