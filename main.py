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

split_sentences = split_into_sentences(
    df, progress_bar=True
)

for i in range(5):
    print('Document id: %s' %split_sentences[0][i])
    print('Sentence: %s \n' %split_sentences[1][i])


# Note that SRL is time-consuming, in particular on CPUs.
# To speed up the annotation, you can also use GPUs via the "cuda_device" argument of the "run_srl()" function.

srl_res = run_srl(
    path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", # pre-trained model
    sentences=split_sentences[1],
    cuda_device=-1,
    progress_bar=True,
)


file = open('persian.txt', 'r')
spacy_stopwords = list(file.read().splitlines())

# NB: This step usually takes several minutes to run. You might want to grab a coffee.

narrative_model = build_narrative_model(
    srl_res=srl_res,
    sentences=split_sentences[1],
    embeddings_type="gensim_keyed_vectors",  # see documentation for a list of supported types
    embeddings_path="glove-wiki-gigaword-300",
    n_clusters=[[10]],
    top_n_entities=100,
    stop_words = spacy_stopwords,
    remove_n_letter_words=1,
    progress_bar=True,
)