---
layout: post
title:  "Kick-start gensim doc2vec similarity queries between unseen documents"
date:   2019-12-29 17:36:30 +0100
categories: gensim doc2vec
---

In this blog post I'm describing one approach on how to get a doc2vec similarity
model off the ground quickly and how to apply document similarity queries between
new incoming documents. (Keywords to add: using unsupervised learning without the need
of labelled data)

The setup is roughly the following:
- build a \'document similarity query engine\', which can take a set of unseen reference
documents and take another set of unseen documents which sould be queried against
the former
- the documents of interest are mainly news articles and blog posts
- no training dataset available at the start

The `gensim` `doc2vec` model, which is based on the paragraph vector paper by
[Le and Mikolov][le-and-mikolov-2014], is a great way to quickly get started
with document similarity queries. It basically offers the training of a vector representation of a
given document (or \'paragraph\'), which approximates a vectorized encapsulation of it's meaning.
The question of similarity between documents is then translated into measuring the distance
between paragraph vectors.

But I had two stumbling blocks in my way of getting started:
1. In order to produce meaningful results, the doc2vec training does demand a sufficiently large and also
fitting (with respect to the documents that will be queried in production) corpus of documents.
As already discussed on other places (add links), it either doesn't make sense to use pre-trained
word embeddings to kick-start a doc2vec model.
2. The gensim doc2vec implementation is currently not designed to perform similarity queries against
unknown documents out of the box.

So, what I will show here is how to
  1. Get a publicly available data set
  2. Train doc2vec models
  3. Project new unseen reference documents into the models vector space
  4. Perform similarity queries with unseen documents against the projected reference documents


### 1. Get a publicly available data set

As already stressed before: your training corpus should be on the one hand similar to
the document you will user later for the similarity queries (so essentially: model prediction and distance measurement)
and on the other have a large enough variance of width of topics, so that it can meaningful place
a broad amount of documents in its vector space.

Since I was considering mainly news articles (and some blog posts), I was looking for a
large dataset of freely available news articles. The [gensim-data][gensim-data] API
is generally a good starting point when looking for publicly available
NLP datasets and also models.

But in my case I found a great news article dataset on kaggle. If you have a
kaggle account, you can download a 640 MB large dataset of 143,000 articles from
15 American publications,
submitted by Andrew Thompson: [kaggle link][little-dataset]. (Or get even the larger, 1.5GB dataset with 204,135 articles from 18 American publications from [components.one][big-dataset])

Once we have downloaded the kaggle dataset, we'll extract and concatenate the article
titles and contents and write it all to a text file for subsequent preprocessing.

{% highlight python %}
import pandas as pd

# Use pandas to load the csv files into dataframes (about 1.2 GB in memory)
df_1 = pd.read_csv('/path/to/all-the-news/articles1.csv', index_col=0)
df_2 = pd.read_csv('/path/to/all-the-news/articles2.csv', index_col=0)
df_3 = pd.read_csv('/path/to/all-the-news/articles3.csv', index_col=0)

# Concatenate the three dataframes into one (results in a df with 142568 rows)
df = pd.concat([df_1, df_2, df_3])
{% endhighlight %}

Now let's peak into the dataframe:
{% highlight python %}
>>> df.columns
Index(['id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content'], dtype='object')
>>> df.iloc[0].title
'House Republicans Fret About Winning Their Health Care Suit - The New York Times'
>>> df.iloc[0].content[]
{% endhighlight %}

For now we are only interested in the article title and content, so we can disregard the other columns.
Additionally we have to do some basic data cleaning...

{% highlight python %}
# For our current analysis we don't need fake news ;)
df = df[df['publication']!='Breitbart']

# Remove news source at the end of some headlines
df['title'] = df['title'].str.split(' - ', 1).str[0]

# Keep only the columns we are interested in and drop Nans
df = df[['title', 'content']]
df = df.dropna()

# Concatenate article title and content in a new column
df['concatenated'] = df['title'] + ' ' + df['content']

# Remove newline characters
df['concatenated'] = df['concatenated'].replace(r'\n', ' ', regex=True)
df['concatenated'] = df['concatenated'].replace(r'\^M', ' ', regex=True)

# Write the 'concatenated' column into a text file with one document per line
# The textfile will be around 620 MB
with open('/path/to/filename_of_output.txt', 'w') as f:
  for document in df['concatenated'].tolist():
    f.write(document + '\n')

{% endhighlight %}

This returns a text document with about 120k articles, each
represented as one string in one line of the text document.

### 2. Train doc2vec models

#### 2.1 Read and preprocess the corpus
To read the corpus from disk we use a function which uses gensim simple_preprocess
method to tokenize the documents. For the training data we put the tokenized
documents in a `doc2vec` `TaggedDocument` class wrapper to store it along with a
tag, while for prediction purposes, we return just tokenized version of a
document.
{% highlight python %}
def read_corpus(fname, tokens_only=False, encoding=None

    with open(fname, encoding=encoding, newline='\n') as f:

        for i, line in enumerate(f):

            tokens = gensim.utils.simple_preprocess(line)

            if tokens_only:
                yield tokens

            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

# this will read the corpus into memory
train_corpus = list(read_corpus('/path/to/filename_of_output.txt'))
{% endhighlight %}

Consider to adjust your preprocessing with a custom function, based on your specific corpus.

#### 2.2 Train models
We will train a few Doc2Vec models to compare their performances and pick the best performing
one for our use case. Doc2Vec offers two different modes to train the paragraph vector,
while both are implemented in a shallow neural network with one hidden and one projection
layer. For a nice introductory tutorial see the already stated [documentation][gensim-doc2vec-tutorial].
The two versions are:
- the Distributed Memory Model of Paragraph Vectors (`PV-DM`)
- the Distributed Bag of Words Model of Paragraph Vectors (`PV-DBOW`)

The first version (`PV-DM`) takes all (ordered) word vectors together with a paragraph vector
into account when predicting the next word of a sliding window within a document.
The second version (`PV-DBOW`) doesn't bother about word vectors and tries to predict a word
in a given context solely through the paragraph vector.

For the training of the doc2vec models I followed the nice walk-through of the
paragraph paper results reproduction on the [gensim website][gensim-repro-paragraph]
and adapted it again for this use case:

{% highlight python %}
import multiprocessing
from collections import OrderedDict

import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1

from gensim.models.doc2vec import Doc2Vec

common_kwargs = dict(
    epochs=20, # commonly used value for doc2vec + large corpora
    min_count=5, # the minimum nr of occurrences of a word to be considered
    workers=multiprocessing.cpu_count(),
)

simple_models = [

    # PV-DBOW plain w/ 20 dimensions
    Doc2Vec(dm=0, vector_size=20, **common_kwargs),

    # PV-DBOW plain w/ 20 dimensions and explicit word-vector training
    Doc2Vec(dm=0, vector_size=20, dbow_words=1, **common_kwargs),

    # PV-DBOW plain w/ 40 dimensions
    Doc2Vec(dm=0, vector_size=40, **common_kwargs),

    # PV-DM w/ averaging of context word vectors (default dm_concat=0)
    # 20 dimensions and a sliding window of 5 words before and after (default)
    Doc2Vec(dm=1, vector_size=20, window=5, **common_kwargs),

    # PV-DM w/ averaging of context word vectors (default dm_concat=0)
    # and 40 dimensions
    Doc2Vec(dm=1, vector_size=40, window=5, **common_kwargs),

    # PV-DM w/ averaging of context word vectors (default dm_concat=0);
    # a higher starting alpha may improve CBOW/PV-DM modes;
    # and choosing a relatively large context window
    Doc2Vec(dm=1, vector_size=20, window=10, alpha=0.05, **common_kwargs),

    # PV-DM w/ concatenation of context word vectors - big, slow, experimental
    Doc2Vec(dm=1, vector_size=20, dm_concat=1, window=5, **common_kwargs)

]

for model in simple_models:
    model.build_vocab(train_corpus)
    print("%s vocabulary scanned & state initialized" % model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)
{% endhighlight %}

#### 2.3 Save models

{% highlight python %}
import datetime

now = datetime.datetime.now()

# Parse the model name to an appropriate file name format
replace_tuples = ('(','_'), (')','_'), (',','_'), (' ','_'), ('/','-')

# Write all models to disk
for model in simple_models:

    model_name = str(model)

    for r in replace_tuples:
        model_name = model_name.replace(*r)

    model_filepath = f'/path/to/models/{model_name}-{now.hour}{now.minute}'

    print(f'Saving model {str(model)} to: \n {model_filepath}')
    model.save(model_filepath)

{% endhighlight %}

#### 2.4 Evaluate models
The first basic sanity check for the freshly trained models also follows the
approach of gensim's Doc2Vec [tutorial][gensim-doc2vec-tutorial]:
we use the models to infer new vectors from the training corpus and  
examine how often they are predicted as most similar to its own representation
as part of the trained paragraph vectors from the training corpus.

{% highlight python %}
import random
import collections

# Pick a random subset of documents from the training corpus to speed up testing
random_idxs = [random.randint(0, len(train_corpus)) for i in range(1000)]


for model in simple_models:

    ranks = []

    for idx in random_idxs:

        # The infer_vector method takes a tokenized document (a list of strings)
        # and projects it into the model's vector space
        inferred_vector = model.infer_vector(train_corpus[idx].words)

        # The most_similar method of model.docvecs returns the most similar
        # paragraph vectors learned from the training data with respect to
        # submitted inferred_vector
        sims = model.docvecs.most_similar([inferred_vector], topn=3)
        rank = [docid for docid, sim in sims]

        if idx in rank:
            rank = [docid for docid, sim in sims].index(idx)

        else:
            rank = 'not in top 3'

        ranks.append(rank)


    counter = collections.Counter(ranks)

    print(f'Model: {str(model)} \n counter: {counter} \n')

{% endhighlight %}

And we see:

{% highlight python %}
Model: Doc2Vec(dbow,d20,n5,mc5,s0.001,t4)
counter: Counter({0: 985, 99: 11, 1: 4})

Model: Doc2Vec(dbow+w,d20,n5,w5,mc5,s0.001,t4)
counter: Counter({0: 972, 99: 14, 1: 14})

Model: Doc2Vec(dbow,d40,n5,mc5,s0.001,t4)
counter: Counter({0: 984, 99: 11, 1: 4, 2: 1})

Model: Doc2Vec(dm/m,d20,n5,w5,mc5,s0.001,t4)
counter: Counter({0: 877, 99: 62, 1: 48, 2: 13})

Model: Doc2Vec(dm/m,d40,n5,w5,mc5,s0.001,t4)
counter: Counter({0: 976, 99: 14, 1: 9, 2: 1})

Model: Doc2Vec(dm/m,d20,n5,w10,mc5,s0.001,t4)
counter: Counter({0: 827, 99: 86, 1: 60, 2: 27})

Model: Doc2Vec(dm/c,d20,n5,w5,mc5,s0.001,t4)
counter: Counter({0: 981, 99: 12, 1: 6, 2: 1})

{% endhighlight %}

The first basic assessment reveals that:
  - more dimensions for a PV-DBOW model have no impact on this performance measure
  - the worst performing model with 82.7% is PV-DM w/ averaging of context word vectors + higher starting alpha + large context window
  - the first overall assessment for ... is good

So we can continue with further testing...

### 3. Project new unseen reference documents into the models vector space
Gensim's `doc2vec` model has an out-of-the-box storage of
all paragraph vectors from the training corpus (`model.docvecs`). But we need
to find a way to store the projected paragraph vectors from new documents.

Based on a SO [hint][SO-subset-vectors] from Gordon Mohr, we'll use the
`WordEmbeddingsKeyedVectors` class of gensim to store the inferred paragraph vectors of
new unseen reference documents. Although that class was written for
word vectors in mind, it can be used as a data structure for paragraph vectors,
since the relevant `most_similar` method calculates a distance between two vectors
based on [cosine similarity][cosine-similarity] - it doesn't matter in this
regard whether the vectors represent word embeddings or paragraphs, as long as
they live in the same vector space.

We'll use gensim's built-in [small Lee corpus][small-lee-corpus] as a set of
new unseen documents:
{% highlight python %}
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

# These are the new unseen documents

reference = list(read_corpus(lee_train_file,
                             encoding="iso-8859-1",
                             tokens_only=True))

# Infer vectors for reference items
inferred_vectors = []

for doc in reference:
    vec = model.infer_vector(doc)
    inferred_vectors.append(vec)

# Construct subset of vectors
subset_vectors = WordEmbeddingsKeyedVectors(vector_size=model.vector_size)
labels = list(range(len(inferred_vectors)))

subset_vectors.add(entities=labels , weights=inferred_vectors)   
{% endhighlight %}

### 4. Perform similarity queries with unseen documents against the projected reference documents
Finally, we will take a set of new documents and compare these against the set
of reference documents, which we stored in section 3. in the `WordEmbeddingsKeyedVectors`
class. This will at the same time function as a visual evaluation on how
meaningful the models similarity predictions appear to us.

{% highlight python %}
for model in simple_models:
    print('##############################################')
    print('Now evaluating: ' + str(model))

    # The reference
    reference = list(read_corpus(lee_train_file,
                                 encoding="iso-8859-1",
                                 tokens_only=True))

    # infer vecs for reference items
    inferred_vectors = []

    for doc in reference:
        vec = model.infer_vector(doc)
        inferred_vectors.append(vec)

    # Construct subset of vecs
    vector_size = model.vector_size

    subset_vectors = WordEmbeddingsKeyedVectors(vector_size=vector_size)
    labels = list(range(len(inferred_vectors)))
    subset_vectors.add(entities=labels, weights=inferred_vectors)   

    # Test files are from Lee as well
    test_corpus = list(read_corpus(lee_test_file,
                                   encoding="iso-8859-1",
                                   tokens_only=True))

    doc_id = 10 # fix a random document id to compare it across all models
    compared_doc = test_corpus[doc_id]
    inferred_vector = model.infer_vector(compared_doc)
    sims = subset_vectors.most_similar([inferred_vector], topn=3)

    print('compared document: ' + ' '.join(compared_doc))
    print()
    for idx, score in sims[:1]:
        print(f'MOST SIMILAR {idx}: w score {score}:')
        print(' '.join(reference[int(idx)]))

{% endhighlight %}

[cosine-similarity]: https://en.wikipedia.org/wiki/Cosine_similarity
[small-lee-corpus]: https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf
[gensim-data]: https://github.com/RaRe-Technologies/gensim-data
[SO-subset-vectors]: https://stackoverflow.com/questions/56130065/how-to-perform-efficient-queries-with-gensim-doc2vec
[gensim-doc2vec-tutorial]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-download-auto-examples-tutorials-run-doc2vec-lee-py
[big-dataset]: https://components.one/datasets/all-the-news-articles-dataset/
[little-dataset]: https://www.kaggle.com/snapcrack/all-the-news/data
[gensim-repro-paragraph]: https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html#sphx-glr-auto-examples-howtos-run-doc2vec-imdb-py
[le-and-mikolov-2014]: https://arxiv.org/pdf/1405.4053.pdf
