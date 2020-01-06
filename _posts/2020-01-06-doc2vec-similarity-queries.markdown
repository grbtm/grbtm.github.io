---
layout: post
title:  "Kick-start gensim doc2vec similarity queries between unseen documents"
date:   2020-01-06 10:36:30 +0100
categories: gensim doc2vec
---

In this blog post I describe one approach to getting a `doc2vec` similarity
model off the ground quickly and how to apply document similarity queries *between*
new incoming documents.

The starting point and goals are roughly the following:
- to build a \'document similarity query engine\', which takes two different sets
of unseen documents and queries them against each other
 <!-- a set of unseen reference
documents and take another set of unseen documents which sould be queried against
the former -->
- the documents are mainly news articles and blog posts
- no training dataset available at the start

The [`gensim`][gensim-home] `doc2vec` model, which is based on the paragraph vector paper by
[Le and Mikolov][le-and-mikolov-2014], is a great way to quickly get started
with document similarity queries. It trains a vector representation of a
given document (or \'paragraph\'), which approximates its basic meaning.
Comparing the similarity between documents then becomes a question of measuring the distance
between paragraph vectors. Another nice thing about the `doc2vec`/paragraph vector
model is that it's an unsupervised learning algorithm and thus doesn't ask for labelled
training data.

But there were two stumbling blocks in getting started:
1. In order to produce meaningful results, training the `doc2vec` model **demands a sufficiently large and also
appropriate** (with respect to the documents that will be queried in production) **corpus of documents**.
[As already discussed in [other][pre-trained-embeddings1] [places][pre-trained-embeddings2], it's not worth it to use pre-trained
word embeddings to kick-start a `doc2vec` (PV-DM) model, but more on that in a bit.]
2. The gensim `doc2vec` implementation is currently **not designed to perform similarity queries against
unknown documents out-of-the-box**. To be clear: you can take any new document
and query it for similarity against the documents used for training, but you can't
query it against a set of unknown documents.

So, what I will show here is how to:
  1. Get a publicly available data set
  2. Train multiple doc2vec models
  3. Project new unseen reference documents into the model's vector space
  4. Run similarity queries with other unseen documents against the projected reference documents


### 1. Get a publicly available data set

As mentioned before, your training corpus should, on the one hand, be similar to
the document you will use later for the similarity queries
and on the other hand have a large enough breadth of topics that it can meaningfully place
a wide spectrum of documents in its vector space.

Since I was considering mainly news articles (and some blog posts), I was looking for a
large dataset of freely available news articles. The [gensim-data][gensim-data] API
is generally a good starting point when looking for publicly available
NLP datasets as well as models.

However, in my case I found a great news article dataset on kaggle. If you have a
kaggle account, you can download a 640 MB large dataset of 143,000 articles from
15 American publications,
submitted by Andrew Thompson: [kaggle link][little-dataset]. (Or even get the larger, 1.5GB dataset with 204,135 articles from 18 American publications from [components.one][big-dataset])

Once you have downloaded the kaggle dataset, you can extract and concatenate the article
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

Now let's take a look into the dataframe:
{% highlight python %}
>>> df.columns
Index(['id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content'], dtype='object')
>>> df.iloc[0].title
'House Republicans Fret About Winning Their Health Care Suit - The New York Times'
>>> df.iloc[0].content[:150]
'WASHINGTON  —   Congressional Republicans have a new fear when it comes to their    health care lawsuit against the Obama administration: They might w'
{% endhighlight %}

For now, we are only interested in the article title and content, so we can disregard the other columns.
Additionally, we have to do some basic data cleaning...

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

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Write the 'concatenated' column into a text file with one document per line
# The text file will be around 620 MB
with open('/path/to/filename_of_output.txt', 'w') as f:
  for document in df['concatenated'].tolist():
    f.write(document + '\n')

{% endhighlight %}

This returns a text document with about 120,000 articles, each
represented as one string in one line of the text document.

### 2. Train doc2vec models

#### 2.1 Read and preprocess the corpus
To read the corpus from disk - and thus go easy on memory consumption - we use
a python generator which utilizes gensim's `simple_preprocess` method to
tokenize the documents. For training data we put the tokenized documents in a
`doc2vec` `TaggedDocument` class wrapper to store it along with a tag.
For prediction purposes, we return just a tokenized version of the document.
{% highlight python %}
def read_corpus(fname, tokens_only=False, encoding=(None):

    with open(fname, encoding=encoding, newline='\n') as f:

        for i, line in enumerate(f):

            tokens = gensim.utils.simple_preprocess(line)

            if tokens_only:

                yield tokens

            else:

                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
{% endhighlight %}

Consider adjusting your preprocessing with a custom function based on your specific corpus.

For model training later we will need an iterator which can be re-started,
therefore the above generator isn't sufficient on its own. We'll wrap it in a class
which returns a fresh generator every time its `__iter__` method is called:
{% highlight python %}
class MyCorpus(object):

  def __init__(self, path, tokens_only=False, encoding=None):
    self.path = path
    self.tokens_only = tokens_only
    self.encoding = encoding

  def __iter__(self):
    return read_corpus(self.path,
                       tokens_only=self.tokens_only,
                       encoding=self.encoding)
{% endhighlight %}
#### 2.2 Train models
We will train a few `doc2vec` models to compare their performances and pick the one that performs best for our use case. `doc2vec` offers two different modes to train the paragraph vector. Both are implemented in a shallow neural network with one hidden and one projection
layer. For a nice introductory tutorial on `doc2vec`, see the already stated [documentation][gensim-doc2vec-tutorial].
The two versions are:
- the Distributed Memory Model of Paragraph Vectors (`PV-DM`)
- the Distributed Bag of Words Model of Paragraph Vectors (`PV-DBOW`)

The first version (`PV-DM`) takes into account all (ordered) word vectors together with a paragraph vector when predicting the next word of a sliding window within a document.
The second version (`PV-DBOW`) doesn't bother about word vectors and tries to predict a word in a given context solely through the paragraph vector.

For the training of the `doc2vec` models I followed the nice walk-through of the
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

# See section 1
corpus_path = '/path/to/filename_of_output.txt'

models = [

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

for model in models:

    # Constructs a dictionary of all unique words in corpus
    # and their respective nr of occurrences
    model.build_vocab(MyCorpus(path=corpus_path))

    print(f"{str(model)} initialized.")

{% endhighlight %}

Let's train:

{% highlight python %}
import time

training_times = []

for model in models:

    print(f"Training {str(model)...}")

    start_time = time.perf_counter()
    model.train(MyCorpus(path=corpus_path),
                total_examples=model.corpus_count,
                epochs=model.epochs)
    end_time = time.perf_counter()
    training_time_mins = (end_time - start_time)/60

    training_times.append((str(model), training_time_mins))

print(f"Done with all training. \n Training times: \n {training_times}")
{% endhighlight %}  

The training times on a machine with an i5 CPU 2.6Ghz x4 and 16GB of RAM are:

{% highlight python %}
Done with all training.
 Training times:
 [('Doc2Vec(dbow,d20,n5,mc5,s0.001,t4)', 20.387654794799893),
 ('Doc2Vec(dbow+w,d20,n5,w5,mc5,s0.001,t4)', 81.36948786924987),
 ('Doc2Vec(dbow,d40,n5,mc5,s0.001,t4)', 20.152021311716574),
 ('Doc2Vec(dm/m,d20,n5,w5,mc5,s0.001,t4)', 30.105442603400054),
 ('Doc2Vec(dm/m,d40,n5,w5,mc5,s0.001,t4)', 31.214435009116762),
 ('Doc2Vec(dm/m,d20,n5,w10,mc5,s0.001,t4)', 30.952987542900033),
 ('Doc2Vec(dm/c,d40,n5,w5,mc5,s0.001,t4)', 42.61115139643322)]
{% endhighlight %}

#### 2.3 Save models

{% highlight python %}
import datetime

now = datetime.datetime.now()

# Parse the model name to an appropriate file name format
replace_tuples = ('(','_'), (')','_'), (',','_'), (' ','_'), ('/','-')

# Write all models to disk
for model in models:

    model_name = str(model)

    for r in replace_tuples:
        model_name = model_name.replace(*r)

    model_filepath = f'/path/to/models/{model_name}-{now.hour}{now.minute}'

    print(f'Saving model {str(model)} to: \n {model_filepath}')
    model.save(model_filepath)

{% endhighlight %}

#### 2.4 Evaluate models
The first basic sanity check for the freshly trained models also follows the
approach of gensim's `doc2vec` [tutorial][gensim-doc2vec-tutorial]:
we use the models to infer new vectors from the training corpus and  
examine how often they are predicted as most similar to their own representation
as part of the trained paragraph vectors from the training corpus.

{% highlight python %}
import random
import collections

# Pick a random subset of documents from the training corpus to speed up testing
random_idxs = [random.randint(0, len(train_corpus)) for i in range(1000)]


for model in models:

    ranks = []

    for idx in random_idxs:

        # The infer_vector method takes a tokenized document (a list of strings)
        # and projects it into the model's vector space
        inferred_vector = model.infer_vector(train_corpus[idx].words)

        # The most_similar method of model.docvecs returns for a given
        # 'inferred_vector' the 'topn' most similar paragraph vectors from
        # the training corpus
        sims = model.docvecs.most_similar([inferred_vector], topn=3)
        rank = [docid for docid, sim in sims]

        if idx in rank:
            rank = [docid for docid, sim in sims].index(idx)

        else:
            rank = 'not in top 3'

        ranks.append(rank)


    counter = collections.Counter(ranks)
    perc = counter[0]/len(random_idxs)
    print(f'Model: {str(model)} \n similar to oneself percentage: {perc} \n',
           'The whole counter: {counter} \n')

{% endhighlight %}

And we see:

{% highlight python %}
Model: Doc2Vec(dbow,d20,n5,mc5,s0.001,t4)
 similar to oneself percentage: 0.994
 The whole counter: Counter({0: 994, 1: 4, 'not in top 3': 1, 2: 1})

Model: Doc2Vec(dbow+w,d20,n5,w5,mc5,s0.001,t4)
 similar to oneself percentage: 0.981
 The whole counter: Counter({0: 981, 1: 13, 2: 4, 'not in top 3': 2})

Model: Doc2Vec(dbow,d40,n5,mc5,s0.001,t4)
 similar to oneself percentage: 0.993
 The whole counter: Counter({0: 993, 1: 7})

Model: Doc2Vec(dm/m,d20,n5,w5,mc5,s0.001,t4)
 similar to oneself percentage: 0.887
 The whole counter: Counter({0: 887, 'not in top 3': 52, 1: 45, 2: 16})

Model: Doc2Vec(dm/m,d40,n5,w5,mc5,s0.001,t4)
 similar to oneself percentage: 0.987
 The whole counter: Counter({0: 987, 1: 11, 2: 2})

Model: Doc2Vec(dm/m,d20,n5,w10,mc5,s0.001,t4)
 similar to oneself percentage: 0.849
 The whole counter: Counter({0: 849, 'not in top 3': 80, 1: 53, 2: 18})

Model: Doc2Vec(dm/c,d40,n5,w5,mc5,s0.001,t4)
 similar to oneself percentage: 0.991
 The whole counter: Counter({0: 991, 1: 7, 2: 2})

{% endhighlight %}

The first basic assessment reveals that:
  - more dimensions for a PV-DBOW model have no impact on this performance measure
  - for a PV-DM model more dimensions correlate with a higher self-reference %
  - the worst-performing model with self-reference percentage of 85% is PV-DM with averaging of context word vectors + higher starting alpha + large context window
  - the first overall assessment for the models with a self-reference percentag above 98% is good

So, we can continue with further testing...

### 3. Project new unseen reference documents into the model's vector space

Gensim's `doc2vec` model offers out-of-the-box similarity queries against
all paragraph vectors from the training corpus (`model.docvecs`). But we want
to be able to perform similarity queries against any new reference documents
while at the same time making use of the model's trained vector space.

Based on a SO [hint][SO-subset-vectors] from Gordon Mohr, we'll use the
`WordEmbeddingsKeyedVectors` class of gensim to store the inferred paragraph vectors of
new unseen reference documents. Although that class was written with
word vectors in mind, it can be used as a data structure for paragraph vectors.
This is because the relevant `most_similar` method calculates a distance between two vectors
based on [cosine similarity][cosine-similarity]. It doesn't matter whether the vectors represent word embeddings or paragraphs as long as
they live in the same vector space.

We'll use gensim's built-in [small Lee corpus][small-lee-corpus] as a set of
new unseen documents:

{% highlight python %}
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

# These are the new unseen documents that we want to use as reference docs
# We are using the same MyCorpus class and thus same preprocessing function which
# is important for consistency
test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
lee_corpus = os.path.join(test_data_dir, 'lee_background.cor')

# It's a small corpus, so we can load it into memory
reference = list(MyCorpus(path=lee_corpus,
                          tokens_only=True,
                          encoding="iso-8859-1"))

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

Now that we have the inferred vectors stored as `WordEmbeddingsKeyedVectors`
we can run similarity queries against them. Just keep in mind that since these
vectors are now independent of the model that they were inferred from, they can't be further
trained in this state. (The `doc2vec` model also currently doesn't offer the option
to train a model with an updated training corpus.)

### 4. Run similarity queries with other unseen documents against the projected reference documents

Finally, we will take a set of new documents and compare these against the set
of reference documents that we stored in section 3 in the `WordEmbeddingsKeyedVectors`
class. This will then function as visual material for us to evaluate how
meaningful the models similarity predictions appear to us.

The new unseen reference documents will be again taken from the Lee corpus.
The other set of new documents that will be queried against the former are
another separate set of 50 documents from Lee, which comes also pre-installed
with gensim.

{% highlight python %}
import pprint

# The reference documents to be queried against (same as in section 3)
reference = list(MyCorpus(path=lee_corpus,
                          tokens_only=True,
                          encoding="iso-8859-1"))

# Test files are another set of 50 different docs from Lee as well
lee_50 = os.path.join(test_data_dir, 'lee.cor')
test_corpus = list(MyCorpus(path=lee_50,
                            tokens_only=True,
                            encoding="iso-8859-1"))


# Choose one random sample from test files to comare across all models
doc_id = random.randint(0, len(test_corpus) - 1)
compared_doc = test_corpus[doc_id]

print('Compared document: \n')
pprint.pprint(' '.join(compared_doc))

for model in models:

    print('Now evaluating: ' + str(model))

    # Infer vectors for reference items
    inferred_vectors = []

    for doc in reference:
        vec = model.infer_vector(doc)
        inferred_vectors.append(vec)

    # Construct subset of reference vectors
    subset_vectors = WordEmbeddingsKeyedVectors(vector_size=model.vector_size)
    labels = list(range(len(inferred_vectors)))
    subset_vectors.add(entities=labels , weights=inferred_vectors)   

    # Take randomly picked document and query its similarity against reference
    inferred_vector = model.infer_vector(compared_doc)
    sims = subset_vectors.most_similar([inferred_vector], topn=3)

    print(f'Most similar document from reference subset: \n',
          f' w score {sims[0][1]}:')
    pprint.pprint(' '.join(reference[sims[0][0]]))

{% endhighlight %}

The sample output for one queried document:

{% highlight python %}
Compared document:

('in malawi as in other countries in the region aids is making the effects of '
 'the famine much worse the overall hiv infection rate in malawi is per cent '
 'but in some areas up to percent of people are infected significant '
 'proportion of the young adult population is too sick to do any productive '
 'work malnutrition causes people to succumb to the disease much more quickly '
 'than they do in the west and hunger forces women into prostitution in order '
 'to feed their families making them more vulnerable to contracting the '
 'disease life expectancy has been reduced to years')

 Now evaluating: Doc2Vec(dbow,d20,n5,mc5,s0.001,t4)

Most similar document from reference subset:
  w score 0.8525593280792236:
('the northern territory aids council says it is not surprised the territory '
 'rate of hiv infection through male to female sex is twice the national rate '
 'report in the centre for disease control bulletin says percent of hiv '
 'infections in the territory were from male to female sex compared to per '
 'cent nationally the council frank farmer says it is another reason for '
 'people to practice safe sex we do need to be reminded think people become '
 'bit complacent and they feel that it can happen to them mr farmer said what '
 'these statistics show is that there is shift in the means of transmission '
 'previous figures were about per cent but male to male contact has turned '
 'around here so that it is bit of wake up call for people')

 Now evaluating: Doc2Vec(dbow+w,d20,n5,w5,mc5,s0.001,t4)

Most similar document from reference subset:
  w score 0.7811766266822815:
('drug education campaigns appear to be paying dividends with new figures '
 'showing per cent drop in drug related deaths last year according to the '
 'australian bureau of statistics people died from drug related causes in the '
 'year that figure is substantial drop from when australians died of drug '
 'related causes across the states and territories new south wales recorded '
 'the biggest decrease the bureau david payne attributes the decline of drug '
 'deaths to the heroin drought in some parts of the country better equipped '
 'ambulances and emergency wards and above all effective federal and state '
 'drug education campaigns they have put lot of money into the program there '
 'has been fall and while you can discern trend from that the figures are '
 'going in the right way right direction mr payne said')

 Now evaluating: Doc2Vec(dbow,d40,n5,mc5,s0.001,t4)

Most similar document from reference subset:
  w score 0.7947083711624146:
('today is world aids day and the latest figures show that million people are '
 'living with hiv world wide the latest united nations report on the aids '
 'epidemic has found eastern europe and the republics of the former soviet '
 'union are becoming the new battleground in the fight against the disease un '
 'officials say in russia the number of people carrying hiv doubles almost '
 'annually while ukraine has become the first nation in europe to report per '
 'cent of its adult population is hiv positive the officials say combination '
 'of economic insecurity high unemployment and deteriorating health services '
 'are behind the steep rise')

 Now evaluating: Doc2Vec(dm/m,d20,n5,w5,mc5,s0.001,t4)

Most similar document from reference subset:
  w score 0.8218603134155273:
('today is world aids day and the latest figures show that million people are '
 'living with hiv world wide the latest united nations report on the aids '
 'epidemic has found eastern europe and the republics of the former soviet '
 'union are becoming the new battleground in the fight against the disease un '
 'officials say in russia the number of people carrying hiv doubles almost '
 'annually while ukraine has become the first nation in europe to report per '
 'cent of its adult population is hiv positive the officials say combination '
 'of economic insecurity high unemployment and deteriorating health services '
 'are behind the steep rise')

 Now evaluating: Doc2Vec(dm/m,d40,n5,w5,mc5,s0.001,t4)

Most similar document from reference subset:
  w score 0.7889090776443481:
('today is world aids day and the latest figures show that million people are '
 'living with hiv world wide the latest united nations report on the aids '
 'epidemic has found eastern europe and the republics of the former soviet '
 'union are becoming the new battleground in the fight against the disease un '
 'officials say in russia the number of people carrying hiv doubles almost '
 'annually while ukraine has become the first nation in europe to report per '
 'cent of its adult population is hiv positive the officials say combination '
 'of economic insecurity high unemployment and deteriorating health services '
 'are behind the steep rise')

 Now evaluating: Doc2Vec(dm/m,d20,n5,w10,mc5,s0.001,t4)

Most similar document from reference subset:
  w score 0.7860583662986755:
('new study shows that nearly one third of the aboriginal and torres strait '
 'islander population in australia have been arrested in the past five years '
 'the study conducted by the australian national university for the new south '
 'wales bureau of crime statistics is the first to compare the arrest rates of '
 'the aboriginal and non aboriginal population it finds that unemployment '
 'alcohol and assault rates were the main causes study author boyd hunter says '
 'policy both on community and government level must deal with these issues if '
 'the arrest rate is to be decreased addressing the supply of alcohol in '
 'remote communities is seen as the most likely avenue for reducing rates of '
 'abuse alcohol abuse and hence reduce arrest rates in those communities he '
 'said')

 Now evaluating: Doc2Vec(dm/c,d40,n5,w5,mc5,s0.001,t4)

Most similar document from reference subset:
  w score 0.6363005638122559:
('the northern territory aids council says it is not surprised the territory '
 'rate of hiv infection through male to female sex is twice the national rate '
 'report in the centre for disease control bulletin says percent of hiv '
 'infections in the territory were from male to female sex compared to per '
 'cent nationally the council frank farmer says it is another reason for '
 'people to practice safe sex we do need to be reminded think people become '
 'bit complacent and they feel that it can happen to them mr farmer said what '
 'these statistics show is that there is shift in the means of transmission '
 'previous figures were about per cent but male to male contact has turned '
 'around here so that it is bit of wake up call for people')


{% endhighlight %}


The first evaluation of a random sample shows meaningful similarity results. In order
to evaluate the similarity performance more systematically, one should use a
labelled similarity dataset to see how the models perform with respect to such
a benchmark.


[gensim-home]: https://radimrehurek.com/gensim/index.html
[pre-trained-embeddings1]: https://groups.google.com/forum/#!topic/gensim/lsvhf7499q4
[pre-trained-embeddings2]: https://stackoverflow.com/questions/27470670/how-to-use-gensim-doc2vec-with-pre-trained-word-vectors/30337118#30337118
[cosine-similarity]: https://en.wikipedia.org/wiki/Cosine_similarity
[small-lee-corpus]: https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf
[gensim-data]: https://github.com/RaRe-Technologies/gensim-data
[SO-subset-vectors]: https://stackoverflow.com/questions/56130065/how-to-perform-efficient-queries-with-gensim-doc2vec
[gensim-doc2vec-tutorial]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-download-auto-examples-tutorials-run-doc2vec-lee-py
[big-dataset]: https://components.one/datasets/all-the-news-articles-dataset/
[little-dataset]: https://www.kaggle.com/snapcrack/all-the-news/data
[gensim-repro-paragraph]: https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html#sphx-glr-auto-examples-howtos-run-doc2vec-imdb-py
[le-and-mikolov-2014]: https://arxiv.org/pdf/1405.4053.pdf
