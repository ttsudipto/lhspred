from .density import compute_positiveness
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np

def get_risk_class_decision_score(model, estimator, X_test, scale=True) :
    y_pred = model.predict(estimator, X_test, scale)
    probas = [compute_positiveness(y) for y in y_pred]
    return probas

def plot_roc_model_cv(model, risk_scores, verbose=False) :
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(model.estimators)) :
        X_test = model.data[model.test_indices[i]]
        risk_test = risk_scores[model.test_indices[i]]
        probas = get_risk_class_decision_score(model, model.estimators[i], X_test)
        fpr, tpr, thresholds = roc_curve(risk_test, probas)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('AUC(CV ' + str(i) + ') = ' + str(roc_auc))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if verbose == True :
        print('\tMean AUC = ' + str(mean_auc) + ' (+/- ' + str(std_auc) + ')')

    #------ Print as CSV ------#
    #heading = ['FPR']
    #for i in range(len(model.estimators)) :
        #heading.append('TPR' + str(i) + ' AUC = ' + str(round(aucs[i], 4)))
    #heading.append('Mean TPR' + ' AUC = ' + str(round(mean_auc, 4)) + ' (+/- ' + str(round(std_auc, 4)) + ')')
    #print('\t'.join(heading))
    #for i in range(100) :
        #row = [mean_fpr[i]]
        #for j in range(len(model.estimators)) :
            #row.append(tprs[j][i])
        #row.append(mean_tpr[i])
        #print('\t'.join(map(str,row)))
    #------ Print as CSV ------#

    return mean_fpr, mean_tpr, mean_auc, std_auc

def plot_roc_model_blind_cv(model, b_data, risk_scores, verbose=False) :
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(model.estimators)) :
        probas = get_risk_class_decision_score(model, model.estimators[i], b_data)
        fpr, tpr, thresholds = roc_curve(risk_scores, probas)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('AUC(CV ' + str(i) + ') = ' + str(roc_auc))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if verbose == True :
        print('\tMean AUC = ' + str(mean_auc) + ' (+/- ' + str(std_auc) + ')')

    #------ Print as CSV ------#
    #heading = ['FPR']
    #for i in range(len(model.estimators)) :
        #heading.append('TPR' + str(i) + ' AUC = ' + str(round(aucs[i], 4)))
    #heading.append('Mean TPR' + ' AUC = ' + str(round(mean_auc, 4)) + ' (+/- ' + str(round(std_auc, 4)) + ')')
    #print('\t'.join(heading))
    #for i in range(100) :
        #row = [mean_fpr[i]]
        #for j in range(len(model.estimators)) :
            #row.append(tprs[j][i])
        #row.append(mean_tpr[i])
        #print('\t'.join(map(str,row)))
    #------ Print as CSV ------#

    return mean_fpr, mean_tpr, mean_auc, std_auc

def plot_roc_model_blind(model, b_data, risk_scores, verbose=False) :
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    print(b_data.shape, risk_scores.shape)

    probas = get_risk_class_decision_score(model, model.total_estimator, b_data)
    fpr, tpr, thresholds = roc_curve(risk_scores, probas)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    if verbose == True :
        print('\tAUC = ' + str(mean_auc))


    #------ Print as CSV ------#
    #heading = ['FPR']
    #heading.append('Mean TPR' + ' AUC = ' + str(round(mean_auc, 4)))
    #print('\t'.join(heading))
    #for i in range(100) :
        #row = [mean_fpr[i]]
        #row.append(mean_tpr[i])
        #print('\t'.join(map(str,row)))
    #------ Print as CSV ------#

    return mean_fpr, mean_tpr, mean_auc
