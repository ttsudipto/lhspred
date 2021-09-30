from sklearn.svm import SVC
from scipy.stats import gaussian_kde
import numpy as np
from scipy.integrate import simps

def print_gausian(data, min, max) :
    """Method to print the density of SVM scores"""
    
    density = gaussian_kde(data)
    xs = np.linspace(min, max, 100)
    density.covariance_factor = lambda : 0.25
    density._compute_covariance()
    densities = density(xs)
    print('CT severity score', 'Density')
    for i in range(len(xs)) :
        print(round(xs[i], 4), round(densities[i], 4))
    return xs, densities

def print_histogram(data) :
    density, xs = np.histogram(data, bins='scott', density=True)
    return xs[1:], density

def split_pos_neg(target, pos_class=1, neg_class=0) :
    """Method to split positive and negative samples"""
    
    pos_indices = []
    neg_indices = []
    
    for i in range(target.shape[0]) :
        if target[i] == pos_class :
            pos_indices.append(i)
        elif target[i] == neg_class :
            neg_indices.append(i)
        else :
            raise ValueError('Error !!! Invalid value present ...')
    
    return pos_indices, neg_indices

def load_density(pos_or_neg) :
    filename = 'output/densities/densities.csv'
    csvfile = open(filename)
    if pos_or_neg == 'pos' :
        cols = [0,2]
    elif pos_or_neg == 'neg' :
        cols = [0,1]
    else :
        raise ValueError('Error !!! Invalid value in \'pos_or_neg\' arguement ...')
    density_function = np.genfromtxt(csvfile, delimiter=',', skip_header=1, usecols=cols)
    return density_function

def compute_positiveness(score) :
    pos_func = load_density('pos')
    if score < pos_func[0,0] :
        return 0.0
    if score > pos_func[pos_func.shape[0]-1,0] :
        return 1.0
    masked_data = np.ma.masked_where((pos_func[:,0] > score), pos_func[:,0])
    score_index = np.argmax(masked_data)
    #print(masked_data)
    #print(score_index)
    #print(pos_func[:score_index+1,0])
    #print(pos_func[:score_index+1,1])
    return simps(pos_func[:score_index+1,1], pos_func[:score_index+1,0])

def compute_negativeness(score) :
    neg_func = load_density('neg')
    if score > neg_func[neg_func.shape[0]-1,0] :
        return 0.0
    if score < neg_func[0,0] :
        return 1.0
    masked_data = np.ma.masked_where((neg_func[:,0] < score), neg_func[:,0])
    score_index = np.argmin(masked_data)
    #print(masked_data)
    #print(score_index)
    #print(neg_func[score_index:,0])
    #print(neg_func[score_index:,1])
    return simps(neg_func[score_index:,1], neg_func[score_index:,0])

def print_histogram(data, target) :
    pos_indices, neg_indices = split_pos_neg(target)
    min = 0
    max = 25
    neg_data = data[neg_indices]
    pos_data = data[pos_indices]
    neg_hist = np.histogram(neg_data, bins=25, range=(min, max))
    pos_hist = np.histogram(pos_data, bins=25, range=(min, max))
    print(neg_hist)
    print(pos_hist)

def execute(data, target, ch) :
    """Driver function"""

    pos_indices, neg_indices = split_pos_neg(target)
    min = 0
    max = 25
    neg_data = data[neg_indices]
    pos_data = data[pos_indices]
    
    if (ch == 1) : ## all density
        xs, densities = print_gausian(data, min, max)
        #xs, densities = print_histogram(des)
    elif (ch == 2) : ## negative density
        xs, densities = print_gausian(neg_data, min, max)
        #xs, densities = print_histogram(neg_des)
    elif (ch == 3) : ## positive density
        xs, densities = print_gausian(pos_data, min, max)
        #xs, densities = print_histogram(pos_des)
    else :
        print('Wrong choice')
    
    #print('---------------------------------------')
    #for i in range(len(xs)) :
        #negativeness = compute_negativeness(xs[i])
        #positiveness = compute_positiveness(xs[i])
        #print(xs[i], negativeness, positiveness, negativeness+positiveness)
    
    return xs, densities
    


