import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

data_dir = '/Users/juanpablocasado/Downloads/20_newsgroups'  # your directory

documents = []
for dirpath, dirnames, filenames in os.walk(data_dir):
    print(f"Looking in directory: {dirpath}")
    for file in filenames:
        print(f"Loading file: {file}")
        with open(os.path.join(dirpath, file), 'r', errors='ignore') as f:
            text = f.read()
            documents.append(text)

print(f"Number of documents loaded: {len(documents)}")

nltk.download('stopwords')

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

# Download wordnet if you haven't already
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def stem_and_lemmatize(words):
    stemmed = [stemmer.stem(word) for word in words]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return lemmatized

documents = [stem_and_lemmatize(doc) for doc in documents]

from gensim.models import Phrases

bigram = Phrases(documents, min_count=5)  # at least 5 occurrences for a pair to be considered a bigram
documents = [bigram[doc] for doc in documents]

# Create a dictionary representation of the documents
dictionary = Dictionary(documents * 100)  # Repeat the single document 100 times

# Filter out words that occur less than 10 documents, or more than 70% of the documents
dictionary.filter_extremes(no_below=10, no_above=0.7)

# Bag-of-words representation of the documents
corpus = [dictionary.doc2bow(doc) for doc in documents * 100]  # Repeat the single document 100 times

print(f"Number of words in the first document in the corpus after adjusting filter_extremes: {len(corpus[0])}")

# Set training parameters
num_topics = 10  # number of topics, this can be tuned later
chunksize = 2000
passes = 20
iterations = 400
eval_every = 1

# Make a index to word dictionary
id2word = dictionary.id2token

# Train the LDA model
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