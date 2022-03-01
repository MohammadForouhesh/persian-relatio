import pandas as pd

# df = load_trump_data("raw")
df = pd.read_excel('clf_status_id.xlsx').sample(100000)
print(df.columns)
df = df[['status_id', 'text']]
df = df.rename(columns={'status_id': 'id', 'text': 'doc'})
print(df.head())

tqdm.pandas()


df.replace('', float('NaN'), inplace=True)
df.replace(' ', float('NaN'), inplace=True)
df.doc = df.doc.progress_apply(lambda item: remove_redundant_characters(remove_emoji(item)))
df.dropna(inplace=True)
