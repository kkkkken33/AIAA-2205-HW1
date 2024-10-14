import pandas as pd

df1 = pd.read_csv('mfcc-66.mlp.csv', header=None)
df2 = pd.read_csv('mfcc-58.mlp.csv', header=None)

total = len(df1)
difference = 0
for i in range(len(df1)):
    id1 = df1.iloc[i, 0]
    id2 = df2.iloc[i, 0]
    character1 = df1.iloc[i, 1]
    character2 = df2.iloc[i, 1]
    if id1 != id2:
        print('Error: id not equal')
        break
    if character1 != character2:
        difference += 1
print('Difference: %d/%d' % (difference, total))