import pandas as pd
import numpy as np

kruskal_df = pd.read_csv('./data/kruskal/kruskal_data_10_15.csv')
lca_df = pd.read_csv('./data/lca/lca_data_10.csv')

combined_df = pd.concat([kruskal_df, lca_df])

shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

shuffled_df.to_csv('./data/mix2/kruskal_lca_data.csv', index=False)
