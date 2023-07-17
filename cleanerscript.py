import os
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel, Phrases

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
with open('number_of_documents.txt', 'w') as f:
    f.write(f"Number of documents loaded: {len(documents)}")

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
with open('clean_documents.txt', 'w') as f:
    for doc in documents:
        f.write(' '.join(doc) + '\n')

nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def stem_and_lemmatize(words):
    stemmed = [stemmer.stem(word) for word in words]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return lemmatized

documents = [stem_and_lemmatize(doc) for doc in documents]
with open('stemmed_and_lemmatized_documents.txt', 'w') as f:
    for doc in documents:
        f.write(' '.join(doc) + '\n')

bigram = Phrases(documents, min_count=5)  # at least 5 occurrences for a pair to be considered a bigram
documents = [bigram[doc] for doc in documents]

dictionary = Dictionary(documents)
dictionary.save('/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/dictionary.pkl')

dictionary.filter_extremes(no_below=10, no_above=0.7)
dictionary.save('/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/filtered_dictionary.pkl')

corpus = [dictionary.doc2bow(doc) for doc in documents]
MmCorpus.serialize('corpus.mm', corpus)

num_topics = 10  # number of topics, this can be tuned later
chunksize = 2000
passes = 20
iterations = 400
eval_every = 1

id2word = dictionary.id2token

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
model.save('/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/lda_model.pkl')
