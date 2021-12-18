import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats.contingency import margins

happinessdf = 0


def getresidual(observed, expected):
    total = observed.sum()
    rsum, csum = margins(observed)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (total - rsum) * (total - csum) / total ** 3
    return (observed - expected) / np.sqrt(v)


def import_data():
    global happinessdf
    happinessdf = pd.read_csv("Americandata.csv")


def calcluateX():
    chi2, p, dof, expected = chi2_contingency(happinessdf.iloc[:, 1:].values)
    print("\n==================chi2================ \n")
    print(chi2)
    print("\n==================p-value================ \n")
    print(p)
    print("\n==================Degree of freedom================ \n")
    print(dof)
    print("\n==================expected================ \n")
    print(expected)
    residual = getresidual(happinessdf.iloc[:, 1:].values, expected)
    print(" \n ================residual================ \n")
    print(residual)


if __name__ == '__main__':
    import_data()
    calcluateX()


"""

==================chi2================ 

936.1394782304601

==================p-value================ 

7.523435335099551e-198

==================Degree of freedom================ 

7

==================expected================ 

[[ 683.71157713  170.92789428  102.55673657   71.7897156    68.37115771 49.91094513   25.63918414   17.09278943]
[1316.28842287  329.07210572  197.44326343  138.2102844   131.62884229 96.08905487   49.36081586   32.90721057]]
 
 ================residual================ 

[[-21.22748525  21.09723883  -8.60193606  13.08843718   1.4785158 10.35475847  -3.60274837   4.17652109]
[ 21.22748525 -21.09723883   8.60193606 -13.08843718  -1.4785158 -10.35475847   3.60274837  -4.17652109]]


"""