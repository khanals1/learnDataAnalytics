import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN


def detectoutliers():
    bankdf = pd.read_csv('bankloan.csv')
    cols = ["x1", "x5", "x6", "x7", "x11", "x13", "x14"]
    for col in cols:
        boxplt = bankdf[col].to_frame().dropna().boxplot()
        plt.title("Boxplot for {0}".format(col))
        plt.show()

    mmscalar = MinMaxScaler()
    loanSeries = mmscalar.fit_transform(bankdf.fillna(0))

    outlier_detection = DBSCAN(min_samples=2, eps=1.13)
    cluster = outlier_detection.fit_predict(loanSeries)
    outliers = bankdf.iloc[(cluster == -1).nonzero()]
    print("Top 10 outliets : ")
    print(outliers)


if __name__ == '__main__':
    detectoutliers()
"""
Top 10 outliets : 
      x1   x2   x3   x4   x5    x6  ...      x14  x15  x16    x17    x18  x19
262    9  NaN  1.0  2.0  3.0  42.0  ...  0.41667    0    0    NaN  0.000    1
845   29  NaN  1.0  2.0  3.0  47.0  ...  0.08333    1    0  1.291    NaN    0
847   29  2.0  0.0  2.0  3.0   NaN  ...  0.00000    1    0  1.291  0.000    0
1129  31  NaN  NaN  NaN  3.0  38.0  ...  0.08333    0    0  0.000  0.000    1
1132  13  3.0  0.0  2.0  1.0   NaN  ...  0.75000    1    1    NaN    NaN    0
1177   8  3.0  0.0  2.0  2.0   NaN  ...  0.33333    1    1  0.000  0.911    0
1247  22  3.0  1.0  NaN  6.0  46.0  ...  0.58333    1    1    NaN  1.949    1
1248  22  NaN  1.0  1.0  NaN  46.0  ...  0.83333    1    1  0.025  1.949    0
1285  19  NaN  NaN  NaN  NaN  28.0  ...  0.41667    1    1    NaN  1.000    1
1304  69  3.0  0.0  1.0  1.0  69.0  ...  1.00000    1    1  0.000  0.437    1
"""