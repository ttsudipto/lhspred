# LHSPred - Lung Health Severity Prediction
A web based tool to predict lung health severity in COVID-19 patients.

It uses *Support Vector Regressor (SVR)* and *Multi-layer Perceptron Regressor (MLPR)* 
trained with COVID-19 patients' data to determine a score 
([CT severity score](http://dibresources.jcbose.ac.in/ssaha4/lhspred/about.html#ctss)) 
that evaluates the involvement of lesions in the lungs. This computed score is then 
used to predict risk of pneumonia.

**Cite as:** 

>Bhattacharjee, S., Saha, B., Bhattacharyya, P., &amp; Saha, S. (2022). 
LHSPred: A web based application for predicting lung health severity. 
*Biomedical signal processing and control*, 77, 103745. 
[https://doi.org/10.1016/j.bspc.2022.103745](https://doi.org/10.1016/j.bspc.2022.103745).

## Using the tool
LHSPred is available at: http://dibresources.jcbose.ac.in/ssaha4/lhspred.

To know more about the methodology, please refer to the 
[About](http://dibresources.jcbose.ac.in/ssaha4/lhspred/about.html) page.

The dataset used by the regression models is available 
[here](http://dibresources.jcbose.ac.in/ssaha4/lhspred/datasets.php?type=tt).
The patient data was originally published by 
[Feng, Z. et al. (2020)](https://doi.org/10.1038/s41467-020-18786-x).

## Development
It is deployed in a Apache HTTPD server. Python libraries used :

* numpy
* scikit-learn (Version-`0.20.0`)
* joblib (Version-`0.14.1`)
* scipy (Version-`1.4.1` for Python3 and version-`1.2.3` for Python2)
* pathlib
* statistics

Currently, trained models of only scikit-learn version-`0.20.0` are saved with both Python2 and Python3.
Plotly JS library is used for density plot in the prediction output.

## Team
* **Sudipto Bhattacharjee** *([ttsudipto@gmail.com](mailto:ttsudipto@gmail.com))*<br/>
  Ph.D. Scholar,<br/>
  Department of Computer Science and Engineering,<br/>
  University of Calcutta, Kolkata, India.<br/>
* **Dr. Banani Saha** *([bsaha_29@yahoo.com](mailto:bsaha_29@yahoo.com))*<br/>
  Associate Professor,<br/>
  Department of Computer Science and Engineering,<br/>
  University of Calcutta, Kolkata, India.
* **Dr. Parthasarathi Bhattacharyya** *([parthachest@yahoo.com](mailto:parthachest@yahoo.com))*<br/>
  Consultant Pulmologist,<br/>
  Institute of Pulmocare and Research,<br/>
  Kolkata, India.
* **Dr. Sudipto Saha** *([ssaha4@jcbose.ac.in](mailto:ssaha4@jcbose.ac.in))*<br/>
  Associate Professor,<br/>
  Division of Bioinformatics,<br/>
  Bose Institute, Kolkata, India.
  
*Please contact Dr. Sudipto Saha regarding any further queries.*
