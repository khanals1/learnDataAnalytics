from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot(df):
    columns = ["discharge_disposition_id", "readmitted", "diabetesMed", "race", "gender", "age"]
    for col in columns:
        sf = (df[col].value_counts(normalize=True) * 100)
        print(sf)
        ind = sf.index
        plt.subplots(figsize=(15, 11))
        plt.pie(sf, labels=ind, radius=1.25, labeldistance=100)
        plt.title(col)
        plt.legend()
        plt.show()
