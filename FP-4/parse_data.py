import pandas as pd
df0 = pd.read_csv('winequality-red.csv')
df0.describe()

df = df0[  ['quality', 'volatile acidity', 'density', 'residual sugar']  ]
df = df.rename( columns={'volatile acidity':'acidity', 'residual sugar':'sugar'} )
df.describe()