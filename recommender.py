import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'ml-100k\u.data', sep='\t', )
print(data.head())