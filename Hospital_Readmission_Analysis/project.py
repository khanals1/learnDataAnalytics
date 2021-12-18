import pandas as pd
import FileUtils
from Final_project import PreProcessing
from Final_project import PatternAnalysis
from Final_project import PredectiveAnalysis
from Final_project import Visualization

pd.set_option('max_columns', None)


class DataProcessing:

    def analytics(self):
        df = FileUtils.getDataframefromFile()
        df = PreProcessing.dropColumn(df)
        cleaned_df = PreProcessing.imputeValues(df, ["admission_type_id", "discharge_disposition_id",
                                                     "admission_source_id", "diag_1", "diag_2", "diag_3",
                                                     "number_diagnoses"])

        cleaned_df.to_csv("clean_df.csv")

        #Visualization.plot(cleaned_df)

        rules = PatternAnalysis.getPattern(cleaned_df)
        readmittedYesRule = (rules[rules['consequents'] == {'readmitted_Yes'}])
        readmittedNoRule = (rules[rules['consequents'] == {'readmitted_No'}])
        readmittedExpiredRule = (rules[rules['consequents'] == {'discharge_disposition_id_Expired'}])

        print("readmittedExpiredRule=>")
        print(readmittedExpiredRule)

        readmittedYesRule.to_csv('readmitted_yes_rules.csv')
        readmittedNoRule.to_csv('readmitted_no_rules.csv')
        readmittedExpiredRule.to_csv('readmitted_expired_rules.csv')

        cleaned_df = PredectiveAnalysis.encodeCols(cleaned_df)


if __name__ == '__main__':
    d = DataProcessing()
    d.analytics()


"""
Support < 0.5

"""