# A Persian Reimplementation of Prof. Eliot Ash's Framework, Relatio
 
> Mohammad H. Forouhesh
> 
> Metodata Inc ®
> 
> April 25, 2022

A Persian Reimplementation of Prof. Eliot Ash's Framework, Relatio
> E. Ash, et al. [Text Semantics Capture Political and Economic Narratives](https://arxiv.org/abs/2108.01720)

# Table of Contents
1. [A Brief Overview](#summary)
2. [Main Problem](#tpa_main)
3. [Illustrative Example](#tpa_example)
4. [I/O](#tpa_io)
5. [Motivation](#tpa_motiv)
6. [Related Works](#tpa_lit)
7. [Contributions of this paper](#tpa_contribution)
8. [Proposed Method](#tpa_method)
9. [Experiments](#tpa_exp)

## A Brief Overview <a name="summary"></a>
  <div style="text-align: justify"> The aim of this project is to quantify latent narrative structures in text documents unsupervised. It maps explicit relations between entity groups and identifies coherent entity groups. In particular, we propose a new method of satisfying this requirement - by identifying who does what to whom and by mapping the relationships and interactions among entities in a corpus. By analysing the Persian Twitter in recent years, our team provides an analysis of political and economic narratives. We demonstrate how narratives are dynamic, sentimental, polarised, and interconnected in the political discourse. </div> 

## Main Problem: <a name="tpa_main"></a>
   <div style="text-align: justify"> Is it possible to represent a list of simple narrative statements as a directed multigraph in which the edges are actions and the nodes are entities? </div> 

   <div style="text-align: justify"> The time interval of analysis in this paper is 1994-2015.  </div>

## Illustrative Example: <a name="tpa_example"></a>
   <div style="text-align: justify"> To determine who said what to whom in a paragraph, the proposed method uses semantic role labelling, which identifies the semantic roles in words. In the case of "Millions of Americans lost their unemployment benefits", the narrative graph would look like this:  </div>
    
    > بازار کریپتو ---هدر داد---> وقت و پول من را

## I/O: <a name="tpa_io"></a>
  * Input: Tweets (textual modality)
  * Output: Predicted salient narratives around historical events.

## Motivation: <a name="tpa_motiv"></a>
1. <div style="text-align: justify"> The articulation of partisan values can be discovered by arranging narratives by relative party usage.  </div> 
2. <div style="text-align: justify"> Narratives provide a new window on polarisation of language in politics. </div>
3. <div style="text-align: justify"> Analysing narratives quantitatively is still largely unexplored. </div>
4. <div style="text-align: justify"> Narrative statements are intuitive and close to the original raw text. </div>
5. <div style="text-align: justify"> Various narratives can form a broader discourse. </div>

## Related (Previous) Works: <a name="tpa_lit"></a>
According to the dependency parser method, it is divided into 3 categories:
1. Dictionary methods rely on matching particular words or phrases [Baker, Bloom and Davis, 2016](https://academic.oup.com/qje/article/131/4/1593/2468873); [Shiller, 2019](https://press.princeton.edu/books/hardcover/9780691182292/narrative-economics); [Enke, 2020](https://www.journals.uchicago.edu/doi/abs/10.1086/708857)
2. Unsupervised learning methods such as topic models and document embeddings break sentences down into words or phrases and ignore grammatical information [Hansen, McMahon and Prat, 2017](https://academic.oup.com/qje/article-abstract/133/2/801/4582916); [Larsen and Thorsrud, 2019](https://ideas.repec.org/a/eee/econom/v210y2019i1p203-218.html); [Bybee et al., 2020](https://www.nber.org/system/files/working_papers/w26648/w26648.pdf).
3. syntactic dependency parsers, which identify grammatical relationships between words [Ash et al., 2020](https://ieeexplore.ieee.org/document/9346539/), will often miss how actors are related in a sentence.

## Contributions of this paper: <a name="tpa_contribution"></a>
1. Multigraph approach that links up entities and their associated actions through a network. 
2. Robustness to word ordering.
3. Giving qualitative researchers a rich context. 
4. This method can provide insights into narrative discourse and polarisation through node centrality and graph distance measures.

## Proposed Method: <a name="tpa_method"></a>
### Stage I: (Feature Extraction) 
Given a sentence, using **Semantic Role Labelling**, *subject* (agent or who), *action* (verb or what) and *target* (patient or whom) of the sentence is extracted. A0, V, and A1 are the corresponding set of phrases. Define the set of all narratives **N ⊂ A0 ⨉ V ⨉ A1 = S**. This set can be too high-dimensional.

### Stage II: (Dimensionality Reduction)
Construct the set E of latent entities such that: (not for verbs)
* |E| < |A0 U A1 |
* N = E ⨉ V ⨉ E

  #### a. Construction
  If an entity is frequent, then it is *explicit*, otherwise, it is *implicit*. Sometimes, a sentence entity has no surface form, this is also categorised under the umbrella term of implicit entity.
  1. **Explicit**: Apply Named Entity Recognition, then chose top L frequent entities. 
  2. **Implicit**: Embed sentence then apply K-Means, each cluster represents a latent entity.

### Stage III: (Pipeline Flowchart)
![ash0](https://user-images.githubusercontent.com/17898264/166262804-d5f74ccc-af92-404e-a076-76560c1ea3f4.png)

### Stage IV: (Narrative Multi-directional Graph)
Find the narrative representation of each sentence, create complex narratives by combination. These complex narratives can be analysed using the degree of input and output.

## Experiments: <a name="tpa_exp"></a>
### Datasets:
U.S. Congressional Record, 1994-2015. Transcripts of speeches made in the House and Senate with names and party affiliations. Often used as a data source for text analysis in social science applications.
### Hyperparams: 
L = 1000
K = 1000

### Results: 
In short narratives capture the following: 
* Historical Events: e.g. Sep 11, war on terror
* Sentiment Polarity
* Partisanship
* Debate Structure

A sample from the output:

![IMG-20220404-WA0001](https://user-images.githubusercontent.com/17898264/166265922-289899fe-f30a-4910-8fc3-8ad3ee8536b4.jpg)
