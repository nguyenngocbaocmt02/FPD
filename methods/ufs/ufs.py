import numpy as np
class UniS:
    def __init__(self):
        pass

    def fit(self):
        pass
        
    def suggest_queries(self, unqueried_index, queries_num):
        # for each iteration
        N_unqueried = len(unqueried_index)
        if N_unqueried == 0:
            return []
        queries = np.random.choice(N_unqueried, min(N_unqueried, queries_num), replace=False)
        return [unqueried_index[query] for query in queries]