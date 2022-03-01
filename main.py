import pandas as pd

# df = load_trump_data("raw")
df = pd.read_excel('/content/drive/MyDrive/Metodata/clf_status_id.xlsx').sample(100000)
print(df.columns)
df = df[['status_id', 'text']]
df = df.rename(columns={'status_id': 'id', 'text': 'doc'})
print(df.head())
