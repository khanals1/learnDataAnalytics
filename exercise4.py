import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import KNNImputer



def imputing(df):
    le = preprocessing.LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    scaler = preprocessing.StandardScaler()
    attributes = [['gender', 'age', 'income', 'tax (15%)']]
    for attribute in attributes:
        df[attribute] = scaler.fit_transform(df[attribute])
    imputer = KNNImputer()
    completedf = imputer.fit_transform(df)
    id_col = list(range(1, 1001))
    cols = ['gender', 'age', 'income', 'tax (15%)']
    finaldf = pd.DataFrame(data=completedf, columns=cols)
    finaldf.index.name = "ID"
    calculating_datapoint(finaldf, "After")



def calculating_datapoint(data, phase):
    mininum_value = data.min().to_string()
    maximum_value = data.max().to_string()
    mean_value = data.mean().to_string()
    median_value = data.median().to_string()
    quartile = data.quantile([.25, .75]).to_string()
    is_nanper_column = data.isna().sum()
    print("################## PRINTING DATA SET " + phase + " imputing #############################")
    print('\n' + 'Minimum Values for Each Column : ')
    print(mininum_value)
    print('\n' + 'Maximum Values for Each Column : ')
    print(maximum_value)
    print('\n' + 'Mean Values for Each Column : ')
    print(mean_value)
    print('\n' + 'Median Values for Each Column : ')
    print(median_value)
    print('\n' + 'Quartiles for each column : ')
    print(quartile)
    print('\n' + 'Nan pair : ')
    print(is_nanper_column)


def correcting_values(df):
    df['gender'] = df['gender'].fillna(pd.Series(np.random.choice(['Male', 'Female'], size=len(df.index))))
    df['gender'] = df['gender'].apply(lambda gender: 'Male' if (gender == 'Man' or gender == 'Men') else (
        'Female' if (gender == 'Women' or gender == 'Woman') else gender))
    df.loc[~(df['age'] > 0), 'age'] = np.NAN
    df.loc[~(df['income'] > 0), 'income'] = np.NAN
    df.loc[~(df['tax (15%)'] > 0), 'tax (15%)'] = np.NAN
    df['income'] = np.where((df.income.isna() & df['tax (15%)'].notnull()), df['tax (15%)'] * 1.15, df.income)
    df['tax (15%)'] = np.where((df.income.notnull() & df['tax (15%)'].isna()), df['income'] * 0.15, df['tax (15%)'])
    df['tax (15%)'] = np.where(
        (df.income.notnull() & df['tax (15%)'].notnull() & df['tax (15%)'] != 0.15 * df['income']),
        round(df['income'] * 0.15, 2),
        df['tax (15%)'])
    calculating_datapoint(df, "Before")
    imputing(df)


def checking_with_rules(df, total_df):
    valid_df = 0
    for row in df.itertuples():
        try:
            data_tax = row[4]
            tax = round(0.15 * int(row.income), 2)
            if row.age > 18 and data_tax == tax and (str(row.income) and row.income != 0.0):
                valid_df = valid_df + 1
        except ValueError:
            pass
    print("Total valid data = " + str(valid_df))
    print("Percentage of valid data = " + str((valid_df / total_df) * 100))
    correcting_values(df)


def checking_complete_data():
    file = "Income Dirty Data.csv"
    df = pd.read_csv(file, index_col=0)
    invalid_df = sum([True for idx, row in df.iterrows() if any(row.isnull())])
    total_df = len(df.index)
    complete_df = total_df - invalid_df
    print("Total complete data = " + str(complete_df))
    print("Percentage of complete data = " + str((complete_df / total_df) * 100))
    checking_with_rules(df, total_df)


if __name__ == '__main__':
    checking_complete_data()

"""
Total complete data = 733
Percentage of complete data = 73.3
Total valid data = 577
Percentage of valid data = 57.699999999999996

################## PRINTING DATA SET Before imputing #############################

Minimum Values for Each Column : 
gender       Female
age            18.0
income         70.0
tax (15%)      10.5

Maximum Values for Each Column : 
gender            Male
age               60.0
income        897055.2
tax (15%)    134558.28

Mean Values for Each Column : 
age             38.964523
income       87035.535066
tax (15%)    13055.330234

Median Values for Each Column : 
age             40.00
income       88139.00
tax (15%)    13220.85

Quartiles for each column : 
       age    income  tax (15%)
0.25  28.0   59504.0    8925.60
0.75  50.0  119909.0   17986.35

Nan pair : 
gender        0
age          98
income       15
tax (15%)    15

################## PRINTING DATA SET After imputing #############################

Minimum Values for Each Column : 
gender      -1.008032
age         -1.661432
income      -1.690137
tax (15%)   -1.690137

Maximum Values for Each Column : 
gender        0.992032
age           1.667055
income       15.742378
tax (15%)    15.742378

Mean Values for Each Column : 
gender       6.217249e-17
age         -1.963199e-02
income      -1.561255e-03
tax (15%)   -1.561256e-03

Median Values for Each Column : 
gender       0.992032
age         -0.036813
income       0.012768
tax (15%)    0.012768

Quartiles for each column : 
        gender       age    income  tax (15%)
0.25 -1.008032 -0.789685 -0.532328  -0.532328
0.75  0.992032  0.795308  0.627473   0.627473

Nan pair : 
gender       0
age          0
income       0
tax (15%)    0
"""
