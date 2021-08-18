from .model import Model

svrParamGridRBF = {
        'kernel' : ['rbf'],
        'gamma' : [1e-1, 1e-2, 1e-3, 1e-4],
        'C' : [5, 10, 15, 20],
        'epsilon' : [0.001, 0.01, 0.1, 0.5, 0.8]
    }
svrParamGridPoly = {
        'kernel' : ['poly'],
        'degree' : [2, 3],
        'gamma' : [1e-1, 1e-2, 1e-3],
        'coef0' : [1, 2, 3, 4],
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

def gridSearchRBF(X, y, k=5) :
    print('kernel', 'C', 'gamma', 'epsilon', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    scale = True
    model = Model('SVR', X, y, k)
    for c in svrParamGridRBF['C'] :
        for g in svrParamGridRBF['gamma'] :
            for e in svrParamGridRBF['epsilon'] :
                param = {'kernel' : 'rbf', 'C' : c, 'gamma' : g, 'epsilon' : e}
                model.learn(param, scale)
                mse, mae, r2s, pcc = model.predict_k_fold(scale)
                print('RBF', c, g, e, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

def gridSearchPoly(X, y, k=5) :
    print('kernel', 'degree', 'C', 'gamma', 'coef0', 'epsilon', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    scale = True
    model = Model('SVR', X, y, k)
    for d in svrParamGridPoly['degree'] :
        for c in svrParamGridPoly['C'] :
            for g in svrParamGridPoly['gamma'] :
                for co in svrParamGridPoly['coef0'] :
                    for e in svrParamGridPoly['epsilon'] :
                        param = {'kernel' : 'poly', 'degree' : d, 'C' : c, 'gamma' : g, 'coef0' : co, 'epsilon' : e}
                        model.learn(param, scale)
                        mse, mae, r2s, pcc = model.predict_k_fold(scale)
                        print('poly', d, c, g, co, e, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))

def gridSearchLinear(X, y, k=5) :
    print('kernel', 'C', 'epsilon', 'MSE', 'MAE', 'R^2-Score', 'PCC')
    scale = True
    model = Model('SVR', X, y, k)
    for c in svrParamGridLinear['C'] :
        for e in svrParamGridLinear['epsilon'] :
            param = {'kernel' : 'linear', 'C' : c, 'epsilon' : e}
            model.learn(param, scale)
            mse, mae, r2s, pcc = model.predict_k_fold(scale)
            print('linear', c, e, round(mse, 3), round(mae, 3), round(r2s, 3), round(pcc, 3))
