---
layout: post
title:  "Kick-start gensim doc2vec similarity queries between unseen documents"
date:   2019-12-29 17:36:30 +0100
categories: gensim doc2vec
---

I had the following problem:

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
large dataset of freely available news articles.

If you have a kaggle account, you can download a 640 MB large dataset of 143,000 articles from 15 American publications,
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
Additionally we will remove the news source at the end of some headlines.

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

# Write the 'concatenated' column into a text file with one document per line
# The textfile will be around 620 MB
with open('/path/to/filename_of_output.txt', 'w') as f:
  for document in df['concatenated'].tolist():
    f.write(document + '\n')

{% endhighlight %}

This returns a text document with about 120k articles and headlines, each
represented as one string in one line of the text document.

### 2. Train doc2vec models

#### 2.1 Read and preprocess the corpus
For the reading of the corpus we follow gensim's [documentation][gensim-doc2vec-tutorial] on `doc2vec` and
slightly adjust it according to our needs:

{% highlight python %}
import smart_open

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname) as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus('/path/to/filename_of_output.txt'))
{% endhighlight %}

Consider to adjust your preprocessing with a custom function, based on your specific corpus.

#### 2.2 Train models and evaluate
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
    # and 20 dimensions
    Doc2Vec(dm=1, vector_size=20, **common_kwargs),

    # PV-DM w/ averaging of context word vectors (default dm_concat=0)
    # and 40 dimensions
    Doc2Vec(dm=1, vector_size=40, **common_kwargs),

    # PV-DM w/ averaging of context word vectors (default dm_concat=0);
    # a higher starting alpha may improve CBOW/PV-DM modes;
    # and choosing a relatively large context window
    Doc2Vec(dm=1, vector_size=20, window=10, alpha=0.05, **common_kwargs),

    # PV-DM w/ concatenation of context word vectors - big, slow, experimental
    # window=5 (both sides) approx. paper's apparent 10-word total window size
    Doc2Vec(dm=1, vector_size=20, dm_concat=1, window=5, **common_kwargs)

]

for model in simple_models:
    model.build_vocab(train_corpus)
    print("%s vocabulary scanned & state initialized" % model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)
{% endhighlight %}

### 3. Project new unseen reference documents into the models vector space


### 4. Perform similarity queries with unseen documents against the projected reference documents

For further discussions on a feasible production setup with even larger and more
frequently changing corpora, see the useful explanations by Gordon Mohr: ADD LINK

<!-- Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight python %}
print('Hi, Tom')
# prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/ -->

[gensim-doc2vec-tutorial]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-download-auto-examples-tutorials-run-doc2vec-lee-py
[big-dataset]: https://components.one/datasets/all-the-news-articles-dataset/
[little-dataset]: https://www.kaggle.com/snapcrack/all-the-news/data
[gensim-repro-paragraph]: https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html#sphx-glr-auto-examples-howtos-run-doc2vec-imdb-py
[le-and-mikolov-2014]: https://arxiv.org/pdf/1405.4053.pdf
