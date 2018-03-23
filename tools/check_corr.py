import pandas as pd
import sys

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])
res = 0
for col in df1.columns.values[1:]: # skip id
    cur = df1[col].corr(df2[col])
    corr = (df1[col].rank() / len(df1)).corr(df2[col].rank() / len(df2))
    print(col, corr)
    res += corr
print("Avg Rank", res / 6)
