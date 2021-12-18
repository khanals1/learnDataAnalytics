import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import statsmodels.formula.api as smf

sales_df = 0


def import_data():
    global sales_df
    sales_df = pd.read_csv("Album Sales 3.csv")


def scatter_plot():
    colors = np.array([0.2, 0.2, 0.2])
    area = np.pi * 4
    pt.scatter(sales_df.adverts, sales_df.sales, s=area, c=colors, alpha=0.5)
    pt.title("Scatter Plot(Sales vs Adverts)")
    pt.xlabel("Advertisement")
    pt.ylabel("Sales")
    pt.show()

    pt.scatter(sales_df.airplay, sales_df.sales, s=area, c=colors, alpha=0.5)
    pt.title("Scatter Plot(Sales vs Airplay)")
    pt.xlabel("Airplay")
    pt.ylabel("Sales")
    pt.show()

    pt.scatter(sales_df.attract, sales_df.sales, s=area, c=colors, alpha=0.5)
    pt.title("Scatter Plot(Sales vs Attract)")
    pt.xlabel("Attract")
    pt.ylabel("Sales")
    pt.show()

    first_linear_model = smf.ols(formula='sales ~ adverts', data=sales_df).fit()
    print("P value of sales and advert = " + first_linear_model.pvalues.to_string())
    print("\n First Model Summary \n")
    print(first_linear_model.summary())
    print("\n First Model Equation \n")
    print(first_linear_model.params.to_string())
    print("\n First Model Prediction for 135,000 \n")
    print(first_linear_model.params[0] + (first_linear_model.params[1] * 135000))

    second_linear_model = smf.ols(formula='sales ~ adverts + airplay + attract', data=sales_df).fit()
    print("\n Second Model P-values \n")
    print(second_linear_model.pvalues.to_string())
    print("\n Second Model Summary \n")
    print(second_linear_model.summary())


if __name__ == '__main__':
    import_data()
    scatter_plot()

"""
P value of sales and advert = 

Intercept    5.967817e-43
adverts      2.941980e-19

First Model Summary 

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  sales   R-squared:                       0.335
Model:                            OLS   Adj. R-squared:                  0.331
Method:                 Least Squares   F-statistic:                     99.59
Date:                Tue, 28 Sep 2021   Prob (F-statistic):           2.94e-19
Time:                        15:14:13   Log-Likelihood:                -1120.7
No. Observations:                 200   AIC:                             2245.
Df Residuals:                     198   BIC:                             2252.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    134.1399      7.537     17.799      0.000     119.278     149.002
adverts        0.0961      0.010      9.979      0.000       0.077       0.115
==============================================================================
Omnibus:                        4.961   Durbin-Watson:                   2.032
Prob(Omnibus):                  0.084   Jarque-Bera (JB):                4.556
Skew:                           0.339   Prob(JB):                        0.102
Kurtosis:                       3.294   Cond. No.                     1.26e+03
==============================================================================

First Model Equation 

Intercept    134.139938
adverts        0.096124

First Model Prediction for 135,000 

13110.945544285487

Second Model P-values 

Intercept    1.266698e-01
adverts      5.054937e-26
airplay      1.326307e-25
attract      9.492121e-06

Second Model Summary 

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  sales   R-squared:                       0.665
Model:                            OLS   Adj. R-squared:                  0.660
Method:                 Least Squares   F-statistic:                     129.5
Date:                Tue, 28 Sep 2021   Prob (F-statistic):           2.88e-46
Time:                        15:14:13   Log-Likelihood:                -1052.2
No. Observations:                 200   AIC:                             2112.
Df Residuals:                     196   BIC:                             2126.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -26.6130     17.350     -1.534      0.127     -60.830       7.604
adverts        0.0849      0.007     12.261      0.000       0.071       0.099
airplay        3.3674      0.278     12.123      0.000       2.820       3.915
attract       11.0863      2.438      4.548      0.000       6.279      15.894
==============================================================================
Omnibus:                        0.610   Durbin-Watson:                   1.950
Prob(Omnibus):                  0.737   Jarque-Bera (JB):                0.351
Skew:                          -0.073   Prob(JB):                        0.839
Kurtosis:                       3.144   Cond. No.                     4.11e+03
==============================================================================













"""
