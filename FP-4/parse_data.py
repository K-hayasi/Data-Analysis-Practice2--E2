import pandas as pd
df0 = pd.read_csv('Real estate valuation data set.csv')
df0.describe()

df = df0[  ['Y house price of unit area', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']  ]
df = df.rename( columns={'Y house price of unit area':'price','X2 house age':'age', 'X3 distance to the nearest MRT station':'from station', 'X4 number of convenience stores':'number of convenience stores'} )
df.describe()