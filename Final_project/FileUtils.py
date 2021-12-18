import pandas as pd


def getDataframefromFile():
    file = "10kPatients.csv"
    df = pd.read_csv(file)
    return df
