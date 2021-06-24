import csv
import numpy as np

np.set_printoptions(threshold=100000)

class DataResource :
    def __init__(self, name) :
        self.path_prefix = 'data/'
        self.fileName = name
        self.attributeMap = self._read_attributes()
        self.data_columns = ['age', 
                             'white blood cell count', 
                             #'neutrophil count', 
                             #'lymphocyte count', 
                             'NLR', 
                             'AST', 
                             'albumin', 
                             'lactic dehydrogenase', 
                             'C-reactive protein'
                             ]
        self.target_column = 'CT severity score'
        self.data = None
        self.target = None
    
    def _read_attributes(self):
        csvfile = open(self.path_prefix + self.fileName, 'r')
        attributes = csv.DictReader(csvfile, delimiter=',').fieldnames
        attributeMap = dict()
        for i in range(len(attributes)) :
            attributeMap[attributes[i]] = i
        csvfile.close()
        return attributeMap
    
    def _converter(self, x) :
        if np.isnan(float(bytes.decode(x))) :
            return 0.
        else :
            return float(x)
    
    def read(self) :
        csvfile = open(self.path_prefix + self.fileName, 'rb')
        data_cols = [self.attributeMap[x] for x in self.data_columns]
        #print(tuple(data_cols))
        self.data = np.genfromtxt(csvfile, delimiter=',', autostrip=True, skip_header=1, usecols=data_cols)
        csvfile.close()
        
        csvfile = open(self.path_prefix + self.fileName, 'rb')
        self.target = np.genfromtxt(csvfile, delimiter=',', autostrip=True, skip_header=1, usecols=self.attributeMap[self.target_column])
        csvfile.close()
        
        self.replaceMissingValues()
        #print(self.data)
        #print(self.target)
    
    def replaceMissingValues(self):
        """Method to perform null value imputation"""
        
        col_mean = np.nanmean(self.data, axis=0)
        x,y = np.where(np.isnan(self.data))
        self._null_indices = [(x[i], y[i]) for i in range(len(x))]
        for i in range(len(x)):
            self.data[x[i],y[i]] = col_mean[y[i]]
    
    def readProgressionData(self) :
        CTScores = []
        outcomes = []
        csvfile = open(self.path_prefix + self.fileName, 'r')
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader :
            CTScores.append(float(row['CT severity score']))
            outcomes.append(int(row['outcome']))
        csvfile.close()
        return np.array(CTScores, dtype=np.float64), np.array(outcomes, dtype=np.int32)

def storeResourceCSV(derivationFileName, validationFileName, trainingFileName, testFileName, testIndices) :
    path_prefix = 'data/'
    derivationFile = open(path_prefix+derivationFileName, 'r')
    validationFile = open(path_prefix+validationFileName, 'r')
    trainingFile = open(path_prefix+trainingFileName, 'w')
    testFile = open(path_prefix+testFileName, 'w')
    
    fieldnames = {'case no.': 'Identifier',
                  'age': 'Age (years)',
                  'white blood cell count': 'White blood cell count - WBC (billions per litre)',
                  'NLR': 'Neutrophil Lymphocyte Ratio - NLR',
                  'AST': 'Aspartate transaminase - AST (U/L)',
                  'albumin': 'Albumin (g/L)',
                  'lactic dehydrogenase': 'Lactic dehydrogenase - LDH (U/L)',
                  'C-reactive protein': 'C-reactive protein - CRP (mg/L)',
                  'CT severity score': 'CT severity score',
                  'outcome': 'Risk'}
    
    derivationReader = csv.DictReader(derivationFile, delimiter=',')
    validationReader = csv.DictReader(validationFile, delimiter=',')
    trainingWriter = csv.DictWriter(trainingFile, fieldnames=fieldnames.values(), delimiter=',')
    testWriter = csv.DictWriter(testFile,fieldnames=fieldnames.values(), delimiter=',')
    trainingWriter.writeheader()
    testWriter.writeheader()
    
    rowNum = 0
    for row in derivationReader :
        if row['outcome'] == '0' :
            row['outcome'] = 'Low'
        else :
            row['outcome'] = 'High'
        outRow = dict()
        for k,v in fieldnames.items() :
            outRow[v] = row[k]
        if rowNum in testIndices :
            testWriter.writerow(outRow)
            #print(str(rowNum) + ' test ' + row['case no.'] + ' ' + row['CT severity score'] + ' ' + row['outcome'])
        else :
            trainingWriter.writerow(outRow)
            #print(str(rowNum) + ' train ' + row['case no.'] + ' ' + row['CT severity score'] + ' ' + row['outcome'])
        rowNum = rowNum + 1
    
    for row in validationReader :
        if row['outcome'] == '0' :
            row['outcome'] = 'Low'
        else :
            row['outcome'] = 'High'
        outRow = dict()
        for k,v in fieldnames.items() :
            outRow[v] = row[k]
        if rowNum in testIndices :
            testWriter.writerow(outRow)
            #print(str(rowNum) + ' test ' + row['case no.'] + ' ' + row['CT severity score'] + ' ' + row['outcome'])
        else :
            trainingWriter.writerow(outRow)
            #print(str(rowNum) + ' train ' + row['case no.'] + ' ' + row['CT severity score'] + ' ' + row['outcome'])
        rowNum = rowNum + 1
    
    derivationFile.close()
    validationFile.close()
    trainingFile.close()
    testFile.close()

#, converters={x:self._converter for x in data_cols}
#, converters={self.attributeMap[self.target_column] : self._converter}
