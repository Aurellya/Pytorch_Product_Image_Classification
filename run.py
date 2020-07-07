import pandas as pd 

df = pd.read_csv('output.csv')
df["filename"] = df["filename"].str.split(pat = "/").str[3]
df.head()

df2 = pd.read_csv('shopee-product-detection-dataset/test.csv')
df2.head()

x =[]
y =[]

for i, row in df2.iterrows():
    temp = df[df['filename'] == row['filename']].category
    temp = temp.reset_index(drop=True)
    x.append(row['filename'])
    y.append(temp[0])

data = {'filename':x, 'category':y}
df2 = pd.DataFrame(data=data)
print(df2.head())
df2.to_csv("final.csv", header = True, index=False)
