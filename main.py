import pandas as pd
import numpy as np

from tqdm import tqdm
from crf_pos.normalization import Normalizer
from src.graphs import build_graph, draw_graph
from src.preprocess import remove_redundant_characters, remove_emoji
from src.utils import split_into_sentences
from src.wrappers import build_narrative_model, run_srl, get_narratives
import os
from src.utils import formalize

tqdm.pandas()

norm = Normalizer()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
df = pd.read_excel('politics.xlsx').sample(100)
df['text'] = df.text.progress_apply(lambda item: formalize(item))
df['text'] = df.text.progress_apply(lambda item: norm.normalize(item))
print(df.columns)
df = df[['status_id', 'text']]
df = df.rename(columns={'status_id': 'id', 'text': 'doc'})
print(df.head())

tqdm.pandas()

## mac version RND
df.replace('', float('NaN'), inplace=True)
df.replace(' ', float('NaN'), inplace=True)
df.doc = df.doc.progress_apply(lambda item: norm.normalize(item))
df.doc = df.doc.progress_apply(lambda item: remove_redundant_characters(remove_emoji(item)))
df.dropna(inplace=True)

split_sentences = split_into_sentences(df, progress_bar=True)

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


import spacy
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# NB: This step usually takes several minutes to run. You might want to grab a coffee.

narrative_model = build_narrative_model(
    srl_res=srl_res,
    sentences=split_sentences[1],
    embeddings_type="gensim_keyed_vectors",  # see documentation for a list of supported types
    embeddings_path="glove-wiki-gigaword-300",
    n_clusters=[[3], [2]],
    top_n_entities=100,
    stop_words=spacy_stopwords,
    remove_n_letter_words=1,
    progress_bar=True,
)

print(narrative_model.keys())

# Most common named entities

print(narrative_model['entities'].most_common()[:20])

# The unnamed entities uncovered in the corpus
# (automatically labeled by the most frequent phrase in the cluster)

final_statements = get_narratives(
    srl_res=srl_res,
    doc_index=split_sentences[0],  # doc names
    narrative_model=narrative_model,
    n_clusters=[0, 0],
    progress_bar=True,
)

# The resulting pandas dataframe

print(final_statements.columns)


# Entity coherence
# Print most frequent phrases per entity

# Pool ARG0, ARG1 and ARG2 together

df1 = final_statements[['ARG0_lowdim', 'ARG0_highdim']]
df1.rename(columns={'ARG0_lowdim': 'ARG', 'ARG0_highdim': 'ARG-RAW'}, inplace=True)

df2 = final_statements[['ARG1_lowdim', 'ARG1_highdim']]
df2.rename(columns={'ARG1_lowdim': 'ARG', 'ARG1_highdim': 'ARG-RAW'}, inplace=True)

df3 = final_statements[['ARG2_lowdim', 'ARG2_highdim']]
df3.rename(columns={'ARG2_lowdim': 'ARG', 'ARG2_highdim': 'ARG-RAW'}, inplace=True)

df = pd.concat([df1, df2, df3]).reset_index(drop=True)

# Count semantic phrases

df = df.groupby(['ARG', 'ARG-RAW']).size().reset_index()
df.columns = ['ARG', 'ARG-RAW', 'count']

# Drop empty semantic phrases

df = df[df['ARG'] != '']

# Rearrange the data

df = df.groupby(['ARG']).apply(lambda x: x.sort_values(["count"], ascending=False))
df = df.reset_index(drop=True)
df = df.groupby(['ARG']).head(10)

df['ARG-RAW'] = df['ARG-RAW'] + ' - ' + df['count'].astype(str)
df['cluster_elements'] = df.groupby(['ARG'])['ARG-RAW'].transform(lambda x: ' | '.join(x))

df = df.drop_duplicates(subset=['ARG'])

df['cluster_elements'] = [', '.join(set(i.split(','))) for i in list(df['cluster_elements'])]

print('Entities to inspect:', len(df))

df = df[['ARG', 'cluster_elements']]

# Low-dimensional vs. high-dimensional narrative statements

# Replace negated verbs by "not-verb"


final_statements['B-V_lowdim_with_neg'] = np.where(final_statements['ARG0_lowdim'] == True,
                                          'not-' + final_statements['B-V_lowdim'],
                                          final_statements['B-V_lowdim'])

final_statements['B-V_highdim_with_neg'] = np.where(final_statements['ARG0_highdim'] == True,
                                           'not-' + final_statements['B-V_lowdim'],
                                           final_statements['B-V_highdim'])

# Concatenate high-dimensional narratives (with text preprocessing but no clustering)

final_statements['narrative_highdim'] = (final_statements['ARG0_highdim'] + ' ' +
                                         final_statements['B-V_highdim_with_neg'] + ' ' +
                                         final_statements['ARG1_highdim'])

# Concatenate low-dimensional narratives (with clustering)

final_statements['narrative_lowdim'] = (final_statements['ARG0_lowdim'] + ' ' +
                                        final_statements['B-V_highdim_with_neg'] + ' ' +
                                        final_statements['ARG1_lowdim'])

# Focus on narratives with a ARG0-VERB-ARG1 structure (i.e. "complete narratives")

indexNames = final_statements[(final_statements['ARG0_lowdim'] == '')|
                             (final_statements['ARG1_lowdim'] == '')|
                             (final_statements['B-V_lowdim_with_neg'] == '')].index

complete_narratives = final_statements.drop(indexNames)


# Plot low-dimensional complete narrative statements in a directed multi-graph

temp = complete_narratives[["ARG0_lowdim", "ARG1_lowdim", "B-V_lowdim"]]
temp.columns = ["ARG0", "ARG1", "B-V"]
temp = temp[(temp["ARG0"] != "") & (temp["ARG1"] != "") & (temp["B-V"] != "")]
temp = temp.groupby(["ARG0", "ARG1", "B-V"]).size().reset_index(name="weight")
temp = temp.sort_values(by="weight", ascending=False)#.iloc[
#     0:100
# ]  # pick top 100 most frequent narratives
temp = temp.to_dict(orient="records")

for l in temp:
    l["color"] = None

G = build_graph(
    dict_edges=temp, dict_args={}, edge_size=None, node_size=10, prune_network=True
)

draw_graph(G, output_filename="persian-twitter.html")