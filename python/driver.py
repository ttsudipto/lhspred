import json
import sys
from statistics import mode
from .input_wrapper import Input
from .pickler import load_model_from_file
from .density import compute_positiveness, compute_negativeness

def convert_to_float(s) :
    if s == '' :
        return 0.0
    else :
        return float(s)

def parse_json(j_string) :
    #print(j_string)
    json_dict = json.loads(j_string)
    inp = Input()
    params = inp.get_all_params()
    for p in params :
        inp.add_value(convert_to_float(json_dict[p]))
    inp.estimator_id = json_dict['model_id']
    return inp

def modelPredict(model, data, dataScaler=None, targetScaler=None) :
    if dataScaler is None or targetScaler is None :
        yPred = model.predict(data)
    else :
        yPred = model.predict(dataScaler.transform(data))
        #transformedTarget = targetScaler.transform(target.reshape(-1,1)).ravel()
        yPred = targetScaler.inverse_transform(yPred)
    
    return yPred

def predict(inp_vector, modelId) :
    model = load_model_from_file(modelId)
    dataScaler, targetScaler = load_model_from_file('Scalers')
    yPred = modelPredict(model, inp_vector, dataScaler=dataScaler, targetScaler=targetScaler)
    pConfidence = compute_positiveness(yPred[0])*100
    sConfidence = compute_negativeness(yPred[0])*100
    output = {'score' : yPred[0], 'positiveness' : pConfidence, 'negativeness' : sConfidence}
    print('JSON-OP>' + json.dumps(output))
    #print(pConfidence, sConfidence)
    return yPred

inp = parse_json(sys.argv[1])
#for i in range(inp.param_length) :
    #print(inp.get_param(i) + ' = ' + str(inp.get_value(i)))
#print(inp.estimator_id)
predict(inp.get_ndarray(), inp.estimator_id)

#from .resource import DataResource
#vres = DataResource('validation.csv')
#vres.read()
#print(vres.data.shape)
#print(vres.target.shape)
#yPredDriver = predict(vres.data, 'MLPR')
#for y in yPredDriver :
    #print(y)
