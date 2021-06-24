from .model import Model

mlprParamGrid1Layer = {
        'activation' : ['relu', 'logistic'],
        'hidden_layer_sizes' : [(50,), (100,), (150,), (200,), (250,), (300,)],
        'learning_rate' : ['constant'],
        'learning_rate_init' : [0.01, 0.001, 0.0001, 0.00001],
        'alpha' : [0.0001],
        #'momentum' : [0.4, 0.7, 0.9]
    }

mlprParamGrid2Layer = {
        'activation' : ['relu', 'logistic'],
        'hidden_layer_sizes' : [(50,10), (100,20), (150,30), (200,40)],
        'learning_rate' : ['constant'],
        'learning_rate_init' : [0.01, 0.001, 0.0001, 0.00001],
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

def gridSearch1Layer(X, y) :
    #mlprParamGrid1Layer = mlprParamGrid1
    print('Activation', 'Learning_strategy', 'Layers', 'Learning_rate', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    scale = True
    model = Model('MLPR', X, y, k=5)
    for a in mlprParamGrid1Layer['activation'] :
        for lr in mlprParamGrid1Layer['learning_rate'] :
            for hls in mlprParamGrid1Layer['hidden_layer_sizes'] :
                for lri in mlprParamGrid1Layer['learning_rate_init'] :
                    param = {
                        'activation' : a,
                        'solver' : 'adam',
                        'random_state' : 1,
                        'max_iter' : 10000,
                        'learning_rate' : lr,
                        'hidden_layer_sizes' : hls,
                        'learning_rate_init' : lri,
                        #'tol' : 1e-5,
                        'alpha' : 0.0001
                    }
                    model.learn(param, scale)
                    mse, mae, r2s, pcc = model.predict_k_fold(scale)
                    print(a, lr, hls, lri, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

def gridSearch2Layer(X, y) :
    print('Activation', 'Learning_strategy', 'Layers', 'Learning_rate', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    #mlprParamGrid2Layer = mlprParamGrid2
    scale = True
    model = Model('MLPR', X, y, k=5)
    for a in mlprParamGrid2Layer['activation'] :
        for lr in mlprParamGrid2Layer['learning_rate'] :
            for hls in mlprParamGrid2Layer['hidden_layer_sizes'] :
                for lri in mlprParamGrid2Layer['learning_rate_init'] :
                    param = {
                        'activation' : a,
                        'solver' : 'adam',
                        'random_state' : 1,
                        'max_iter' : 10000,
                        'learning_rate' : lr,
                        'hidden_layer_sizes' : hls,
                        'learning_rate_init' : lri,
                        'tol' : 1e-4,
                        'alpha' : 0.0001
                    }
                    model.learn(param, scale)
                    mse, mae, r2s, pcc = model.predict_k_fold(scale)
                    print(a, lr, hls, lri, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))
