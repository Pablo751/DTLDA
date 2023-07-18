import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
import gensim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel, Phrases

data_dir = '/Users/TomKobes/Documents/Unibo/Courses/Digital Text/LDA/DOCS'  # your directory

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

def stem_and_lemmatize(words):
    stemmed = [stemmer.stem(word) for word in words]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return lemmatized

documents = [stem_and_lemmatize(doc) for doc in documents]

# Debug: Print the first stemmed and lemmatized document
#print(f"First stemmed and lemmatized document: {documents[0]}")

bigram = Phrases(documents, min_count=5)  # at least 5 occurrences for a pair to be considered a bigram
documents = [bigram[doc] for doc in documents]


# Debug: Print the first document with bigrams
#print(f"First document with bigrams: {documents[0]}")

# Create a dictionary representation of the documents
dictionary = gensim.corpora.Dictionary(documents)

#print(f"Initial dictionary: {dictionary}")

    # Filter out words that occur less than 10 documents, or more than 70% of the documents
    #dictionary.filter_extremes(no_below=10, no_above=0.7)
    #print(f"Dictionary after filtering extremes: {dictionary}")

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


# Set training parameters
num_topics = 10  # number of topics, this can be tuned later
chunksize = 2000
passes = 20
iterations = 400
eval_every = 1

print(corpus)

print(id2word)
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

# Print the topics learned by the model
print("Topics learned by the LDA model:")
topics = model.print_topics()
for topic in topics:
    print(topic)
