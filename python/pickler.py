from joblib import dump, load
from pathlib import Path
from sys import version as pythonVersion
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.preprocessing import StandardScaler

pathPrefix = 'output/models/'+pythonVersion[0]+'/'

def get_version() :
    import sklearn
    #print(sklearn.__version__)
    return sklearn.__version__+'/'

def performScaling(data, target, store = False) :
    dataScaler = StandardScaler()
    transformedData = dataScaler.fit_transform(data)
    targetScaler = StandardScaler()
    transformedTarget = targetScaler.fit_transform(target.reshape(-1,1)).ravel()
    if store == True :
        dump_model_to_file(dataScaler, pathPrefix+get_version()+'Scalers/', 'dataScaler.joblib')
        dump_model_to_file(targetScaler, pathPrefix+get_version()+'Scalers/', 'targetScaler.joblib')
    return dataScaler, targetScaler, transformedData, transformedTarget

def dump_model_to_file(model, prefix, filename = None) :
    if not Path(prefix).exists() :
        Path(prefix).mkdir(parents=True)
    dump(model, prefix + filename, protocol=2)

def load_model_from_file(modelType) :
    if modelType == 'Scalers' :
        dsFilename = pathPrefix+get_version()+'Scalers/dataScaler.joblib'
        tsFilename = pathPrefix+get_version()+'Scalers/targetScaler.joblib'
        dataScaler = load(dsFilename)
        targetScaler = load(tsFilename)
        #print(dsFilename)
        #print(tsFilename)
        return dataScaler, targetScaler
    
    if modelType == 'SVR' :
        filename = pathPrefix+get_version()+'SVR/model.joblib'
    elif modelType == 'MLPR' :
        filename = pathPrefix+get_version()+'MLPR/model.joblib'
    else : 
        raise ValueError('Error !!! Invalid model type arguement passed ...')
    
    model = load(filename)
    #print(filename)
    return model

def saveModelSVR(modelParams, data, target, storeScalers = False) :
    dataScaler, targetScaler, transformedData, transformedTarget = performScaling(data, target, store=storeScalers)
    svr = SVR(**modelParams)
    svr.fit(transformedData, transformedTarget)
    prefix = pathPrefix + get_version()
    dump_model_to_file(svr, pathPrefix+get_version()+'SVR/', 'model.joblib')

def saveModelMLPR(modelParams, data, target, storeScalers = False) :
    dataScaler, targetScaler, transformedData, transformedTarget = performScaling(data, target, store=storeScalers)
    mlpr = MLPR(**modelParams)
    mlpr.fit(transformedData, transformedTarget)
    prefix = pathPrefix + get_version()
    dump_model_to_file(mlpr, pathPrefix+get_version()+'MLPR/', 'model.joblib')

def testModel(modelId, modelParams, data, target, vdata, vtarget) :
    dataScaler, targetScaler, transformedData, transformedTarget = performScaling(data, target, store=False)
    transformedVData = dataScaler.transform(vdata)
    transformedVTarget = targetScaler.transform(vtarget.reshape(-1,1)).ravel()
    if modelId == 'SVR' :
        model = SVR(**modelParams)
    elif modelId == 'MLPR' :
        model = MLPR(**modelParams)
    model.fit(transformedData, transformedTarget)
    yPred = model.predict(transformedVData)
    yPred = targetScaler.inverse_transform(yPred)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    mse = mean_squared_error(vtarget, yPred, multioutput='uniform_average')
    mae = mean_absolute_error(vtarget, yPred, multioutput='uniform_average')
    r2s = r2_score(vtarget, yPred, multioutput='uniform_average')
    pcc, pv = pearsonr(vtarget, yPred)
    print(str(mse), str(mae), str(r2s), str(pcc))
    #print(transformedVTarget)
    #for i in range(vtarget.shape[0]) :
        #print(vtarget[i], yPred[i])
