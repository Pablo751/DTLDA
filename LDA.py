#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # Needed for POS tagging
import re
import gensim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel, Phrases, TfidfModel, CoherenceModel
from nltk.corpus import wordnet
import pyLDAvis.gensim_models
from nltk import pos_tag

if __name__ == '__main__':   
    data_dir = '/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/cleaned3'  # your directory

    documents = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        print(f"Looking in directory: {dirpath}")
        for file in filenames:
            print(f"Loading file: {file}")
            with open(os.path.join(dirpath, file), 'r', errors='ignore') as f:
                text = f.read()
                documents.append(text)

    #print(f"Number of documents loaded: {len(documents)}")

    # Debug: Print the first document
    #print(f"First document: {documents[0]}")

    def clean_document(doc):
        # Remove punctuation
        text = re.sub('[^a-zA-Z]', ' ', doc)
        # Convert to lowercase
        text = text.lower()
        # Split into words
        words = text.split()
        # Remove stopwords
        words = [w for w in words if not w in set(stopwords.words('english'))]
        return words

    documents = [clean_document(doc) for doc in documents]

    # Debug: Print the first cleaned document
    #print(f"First cleaned document: {documents[0]}")

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def lemmatize(words):
        lemmatizer = WordNetLemmatizer()
        tagged_words = pos_tag(words)
        lemmatized = [lemmatizer.lemmatize(word) for word, pos in tagged_words if pos.startswith('N') or pos.startswith('J')]
        return lemmatized

    documents = [lemmatize(doc) for doc in documents]

    # Debug: Print the first stemmed and lemmatized document
    #print(f"First stemmed and lemmatized document: {documents[0]}")

    bigram = Phrases(documents, min_count=5)
    trigram = Phrases(bigram[documents], min_count=3)

    documents = [trigram[bigram[doc]] for doc in documents]

    # Debug: Print the first document with bigrams
    #print(f"First document with bigrams: {documents[0]}")

    # Create a dictionary representation of the documents
    dictionary = gensim.corpora.Dictionary(documents)

    #print(f"Initial dictionary: {dictionary}")

    # Filter out words that occur less than 10 documents, or more than 70% of the documents
    dictionary.filter_extremes(no_below=10, no_above=0.8)
    print(f"Dictionary after filtering extremes: {dictionary}")

    # Bag-of-words representation of the documents
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Debug: Print the first BoW representation
    print(f"First BoW representation: {corpus[0]}")

    # Invert the word-to-id mapping to get id2word
    id2word = {v: k for k, v in dictionary.token2id.items()}

    # Set training parameters
    num_topics = 10  # number of topics, this can be tuned later
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = 1
    max_coherence = -1
    best_model = None
    best_num_topics = None

    # Train the LDA model
    for num_topics in range(2, 21):  # try num_topics from 2 to 20
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )

        # Compute c_v coherence score
        coherence_model_cv = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
        coherence_cv = coherence_model_cv.get_coherence()

        if coherence_cv > max_coherence:
            max_coherence = coherence_cv
            best_model = model
            best_num_topics = num_topics

    print(f'Number of topics: {num_topics}, Coherence Score: {coherence_cv}')

    print(f'Best Model: {best_num_topics} topics, Coherence Score: {max_coherence}')

    # Print the topics learned by the model
    print("Topics learned by the LDA model:")
    topics = model.print_topics()
    for topic in topics:
        print(topic)

    with open('/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/topics.txt', 'w') as f:
        for topic in topics:
            f.write(str(topic) + '\n')

    with open('/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/topics_per_document.txt', 'w') as f:
        for idx, doc in enumerate(documents):
            doc_bow = dictionary.doc2bow(doc)
            doc_topics = model.get_document_topics(doc_bow)
            f.write(f"Document {idx}: {doc_topics}\n")

    # Compute u_mass coherence score
    coherence_model_umass = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='u_mass')
    coherence_umass = coherence_model_umass.get_coherence()
    print('Coherence Score (u_mass): ', coherence_umass)

    # Compute c_v coherence score
    coherence_model_cv = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()
    print('Coherence Score (c_v): ', coherence_cv)

    print("Script completed successfully.")
    



# In[3]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(best_model, corpus, dictionary)
vis


# In[ ]:




