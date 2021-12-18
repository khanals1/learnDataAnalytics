import numpy as np



def dropColumn(df):
    columns_to_remove = []
    df = df.replace("?", np.NaN)
    df = df.replace("None", np.NaN)
    df = df.replace("Not Available", np.NaN)
    df = df.replace("Not Mapped", np.NaN)
    columns = df.columns
    for column in columns:
        invalid_df = df[column].isnull().sum()
        total_df = len(df.index)
        complete_df = total_df - invalid_df
        print("Total complete data for " + column + " " + str(complete_df))
        percentage_of_complete_data = (complete_df / total_df) * 100
        print("Percentage of complete data = " + str(percentage_of_complete_data) + "\n\n")
        if percentage_of_complete_data < 50:
            columns_to_remove.append(column)
    print("Columns with more than 50% missing data : " + str(columns_to_remove) + "\n")
    for delete_column in columns_to_remove:
        del df[delete_column]
        print(delete_column + " removed from data frame.")
    return df


def imputeValues(df, columns):
    for column in columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df

