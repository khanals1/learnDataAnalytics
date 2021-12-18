import pandas as pd
from apyori import apriori


def mining():
    titanic_df = pd.read_csv("TitanicData.csv")

    a_rule = apriori(titanic_df.values, min_support=0.005, min_confidence=0.8, min_lift=2)
    a_rule = list(a_rule)

    filterd_result = []
    for result in a_rule:
        for entry in result.ordered_statistics:
            if 'Yes' in entry.items_add:
                filterd_result.append(entry)

    sorted_result = sorted(filterd_result, key=lambda x: x.lift, reverse=True)

    for results in sorted_result:
        print(str(results))


if __name__ == '__main__':
    mining()


"""
OrderedStatistic(items_base=frozenset({'2nd', 'Child'}), items_add=frozenset({'Yes'}), confidence=1.0, lift=3.09563994374121)
OrderedStatistic(items_base=frozenset({'2nd', 'Female', 'Child'}), items_add=frozenset({'Yes'}), confidence=1.0, lift=3.09563994374121)
OrderedStatistic(items_base=frozenset({'1st', 'Female'}), items_add=frozenset({'Yes'}), confidence=0.9724137931034481, lift=3.0102429797759345)
OrderedStatistic(items_base=frozenset({'Adult', '1st', 'Female'}), items_add=frozenset({'Yes'}), confidence=0.9722222222222221, lift=3.0096499453039534)
OrderedStatistic(items_base=frozenset({'Female', 'Crew'}), items_add=frozenset({'Yes', 'Adult'}), confidence=0.8695652173913044, lift=2.9264725435447416)
OrderedStatistic(items_base=frozenset({'2nd', 'Female'}), items_add=frozenset({'Yes'}), confidence=0.8773584905660378, lift=2.7159859883767217)
OrderedStatistic(items_base=frozenset({'Female', 'Crew'}), items_add=frozenset({'Yes'}), confidence=0.8695652173913044, lift=2.6918608206445307)
OrderedStatistic(items_base=frozenset({'Adult', 'Female', 'Crew'}), items_add=frozenset({'Yes'}), confidence=0.8695652173913044, lift=2.6918608206445307)
OrderedStatistic(items_base=frozenset({'Adult', '2nd', 'Female'}), items_add=frozenset({'Yes'}), confidence=0.8602150537634409, lift=2.6629160806375998)
"""