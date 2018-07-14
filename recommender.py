#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

#data = pd.read_csv(r'ml-100k\u.data', sep='\t', )
#print(data.head())

import numpy as np

from lightfm.datasets import fetch_movielens

data = fetch_movielens(min_rating=5.0)

print(repr(data['train']))
print(repr(data['test']))

from lightfm import LightFM

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

from lightfm.evaluation import precision_at_k

print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())