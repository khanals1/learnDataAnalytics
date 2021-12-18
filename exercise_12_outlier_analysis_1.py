import pandas as pd


def detectoutliers():
    cardf = pd.read_csv("carsDataset.csv")
    factor = 2
    cols = ['mpg', 'qsec', 'hp']
    outliers = {}
    for col in cols:
        up = cardf[col].mean() + cardf[col].std() * factor
        low = cardf[col].mean() - cardf[col].std() * factor
        outliers[col] = cardf[(cardf[col] >= up) | (cardf[col] <= low)]
        print("\n")
        print("Outliers for " + col + "\n")
        print(outliers[col])


if __name__ == '__main__':
    detectoutliers()


"""
Outliers for mpg

     cars.features   mpg  cyl  disp  hp  drat     wt   qsec  vs  am  gear  carb
17        Fiat 128  32.4    4  78.7  66  4.08  2.200  19.47   1   1     4     1
19  Toyota Corolla  33.9    4  71.1  65  4.22  1.835  19.90   1   1     4     1


Outliers for qsec

  cars.features   mpg  cyl   disp  hp  drat    wt  qsec  vs  am  gear  carb
8      Merc 230  22.8    4  140.8  95  3.92  3.15  22.9   1   0     4     2


Outliers for hp

    cars.features   mpg  cyl   disp   hp  drat    wt  qsec  vs  am  gear  carb
30  Maserati Bora  15.0    8  301.0  335  3.54  3.57  14.6   0   1     5     8

"""