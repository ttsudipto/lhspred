from joblib import dump, load
from pathlib import Path
from sys import version as pythonVersion
from .model import Model

pathPrefix = 'output/models/'+pythonVersion[0]+'/'

def get_version() :
    import sklearn
    #print(sklearn.__version__)
    return sklearn.__version__+'/'

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

def saveModel(modelId, modelParams, data, target, storeScalers=True) :
    model = Model(modelId, data, target, k=5)
    model.learn(modelParams, scale=True)
    prefix = pathPrefix+get_version()+modelId+'/'
    dump_model_to_file(model.total_estimator, prefix, 'model.joblib')
    if storeScalers == True :
        dump_model_to_file(model.dataScaler, pathPrefix+get_version()+'Scalers/', 'dataScaler.joblib')
        dump_model_to_file(model.targetScaler, pathPrefix+get_version()+'Scalers/', 'targetScaler.joblib')

def testModel(modelId, modelParams, data, target, vdata, vtarget) :
    model = Model(modelId, data, target, k=5)
    model.learn(modelParams, scale=True)
    mseB, maeB, r2sB, pccB = model.predict_blind_without_CV(vdata, vtarget, scale=True)
    print(mseB, maeB, r2sB, pccB)
