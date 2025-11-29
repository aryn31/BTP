import pandas as pd

df=pd.read_csv("single_summary.csv")

mid=len(df)//2

df1=df.iloc[:mid]
df2=df.iloc[mid:]

#A is the server in FL
# Save both halves
df1.to_csv("dataset_B.csv", index=False)
df2.to_csv("dataset_C.csv", index=False)

print("Dataset successfully split and saved!")
