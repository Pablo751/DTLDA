#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Setup logging
import logging

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Text Processing
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import pos_tag
from textblob import TextBlob

# Word Cloud
from wordcloud import WordCloud

# Topic Modeling
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, Phrases, TfidfModel, CoherenceModel
import pyLDAvis.gensim_models


# Define the directory where the corpus is located
source_dir = '/Users/juanpablocasado/Downloads/OneDrive_1_7-8-2023/txt'  #Source directory of all the original documents in txt format
output_dir = '/Users/juanpablocasado/Downloads/OneDrive_1_7-8-2023/cleanedtxt'  #Output directory of all the cleaned documents in txt format


# Define the cleaning function without lemmatization
def clean_text_without_lemmatization(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Only keeping alphabets and spaces
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

# Define the cleaning function with lemmatization
def get_wordnet_pos(treebank_tag):
    """Map treebank POS tag to first character used by WordNetLemmatizer"""
    tag = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }.get(treebank_tag[0], wordnet.NOUN)
    return tag

def clean_text_with_lemmatization(text):
    words = clean_text_without_lemmatization(text)  # Using the tokenization from the newer script
    lemmatizer = WordNetLemmatizer()
    tagged_words = pos_tag(words)
    # Only keep nouns (starting with 'N') and adjectives (starting with 'J')
    words = [lemmatizer.lemmatize(word) for word, pos in tagged_words if pos.startswith('N') or pos.startswith('J')]
    return words

# Load and clean the corpus based on the cleaning function provided
def load_and_clean_corpus_as_documents(directory_path, cleaning_function):
    all_documents = []
    for file_name in os.listdir(directory_path):
        with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            document_words = cleaning_function(content)
            all_documents.append(document_words)
    return all_documents

# Display the word frequency distribution graph
def plot_word_frequency(words, top_n=30, title_suffix=""):
    freq_dist = FreqDist(words)
    plt.figure(figsize=(15,6))
    freq_dist.plot(top_n, title=f"Top {top_n} Most Common Words {title_suffix}")
    plt.show()

# Load and clean the corpus without lemmatization
all_words_without_lemmatization = load_and_clean_corpus(source_dir, clean_text_without_lemmatization)

# Plot the word frequency distribution for cleaned but not lemmatized text
plot_word_frequency(all_words_without_lemmatization, title_suffix="After Cleaning (No Lemmatization)")

# Load and clean the corpus with lemmatization
all_words_with_lemmatization = load_and_clean_corpus(source_dir, clean_text_with_lemmatization)

# Plot the word frequency distribution for cleaned and lemmatized text
plot_word_frequency(all_words_with_lemmatization, title_suffix="After Cleaning and Lemmatization")

# Generate and display a word cloud
def generate_word_cloud(words, title):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generate word cloud for cleaned but not lemmatized text
generate_word_cloud(all_words_without_lemmatization, title="Word Cloud After Cleaning (No Lemmatization)")

# Generate word cloud for cleaned and lemmatized text
generate_word_cloud(all_words_with_lemmatization, title="Word Cloud After Cleaning and Lemmatization")


# In[28]:


#Save the cleaned texts

for dirpath, dirnames, filenames in os.walk(source_dir):
    structure = os.path.join(output_dir, os.path.relpath(dirpath, source_dir))
    if not os.path.isdir(structure):
        os.mkdir(structure)
    for file in filenames:
        if file.endswith('.pdf'):
            pdf = PdfReader(os.path.join(dirpath, file))
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        elif file.endswith('.txt'):
            with open(os.path.join(dirpath, file), 'r', errors='ignore') as f:
                text = f.read()
        else:
            print(f"Skipping file {file} due to unknown format.")
            continue
        cleaned_text_words = clean_text_with_lemmatization(text)  # Using the lemmatization function from the first snippet
        cleaned_text = ' '.join(cleaned_text_words)
        with open(os.path.join(structure, file.rsplit('.', 1)[0] + '.txt'), 'w') as f:
            f.write(cleaned_text)


# In[29]:


# LDA Model setup 

save_path = '/Users/juanpablocasado/Downloads/OneDrive_1_7-8-2023/samples'  
source_dir = '/Users/juanpablocasado/Downloads/OneDrive_1_7-8-2023/cleanedtxt' 
documents = load_and_clean_corpus_as_documents(source_dir, clean_text_with_lemmatization)

lemmatizer = WordNetLemmatizer()

bigram = Phrases(documents, min_count=5)
trigram = Phrases(bigram[documents], min_count=3)
documents = [trigram[bigram[doc]] for doc in documents]
# Save the first document after bigrams and trigrams
with open(os.path.join(save_path, "sample_after_bigrams_trigrams.txt"), "w") as f:
    f.write(" ".join(documents[0]))



# Create a dictionary representation of the documents
dictionary = gensim.corpora.Dictionary(documents)

# Save Initial Dictionary
with open(os.path.join(save_path, "initial_dictionary.txt"), "w") as f:
    for k, v in dictionary.items():
        f.write(f"{k}: {v}\n")

# Filter out words that occur less than 10 documents, or more than 70% of the documents
dictionary.filter_extremes(no_below=10, no_above=0.8)

# Save Filtered Dictionary
with open(os.path.join(save_path, "filtered_dictionary.txt"), "w") as f:
    for k, v in dictionary.items():
        f.write(f"{k}: {v}\n")

# Bag-of-words representation of the documents
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Save First Few BoW Representations
with open(os.path.join(save_path, "bow_representation.txt"), "w") as f:
    for doc_bow in corpus[:5]:  # Saving the first 5 BoW representations
        f.write(f"{doc_bow}\n")

# Invert the word-to-id mapping to get id2word
id2word = {v: k for k, v in dictionary.token2id.items()}


# In[ ]:


logging.basicConfig(filename='/Users/juanpablocasado/Downloads/OneDrive_1_7-8-2023/samples/logfile.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Train the LDA model with optimized configurations
for num_topics in tqdm(range(25, 41)):  # Loop from 25 to 40 topics, this was defined after training the model with 1 - 50 topics. The higher coherence score comes in this bracket
    for alpha in alphas:
        for eta in etas:
            for chunksize in chunksizes:
                model = LdaModel(
                    corpus=corpus,
                    id2word=id2word,
                    chunksize=chunksize,
                    alpha=alpha,
                    eta=eta,
                    iterations=iterations,
                    num_topics=num_topics,
                    passes=passes,
                    eval_every=eval_every
                )

                # Compute c_v coherence score
                coherence_model_cv = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
                coherence_cv = coherence_model_cv.get_coherence()

                # Compute u_mass coherence score (no need to store it if you just want to print it)
                coherence_model_umass = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='u_mass')
                coherence_umass = coherence_model_umass.get_coherence()
                
                # Printing coherence scores for the current model configuration
                print(f'Num Topics: {num_topics}, Alpha: {alpha}, Eta: {eta}, Chunksize: {chunksize}, Coherence (c_v): {coherence_cv}, Coherence (u_mass): {coherence_umass}')

                if coherence_cv > max_coherence:
                    max_coherence = coherence_cv
                    best_model = model
                    best_num_topics = num_topics

print(f'Best Model: {best_num_topics} topics, Coherence Score (c_v): {max_coherence}')


# Save the best model
best_model.save("/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/best_model")


# Print the topics learned by the best model
print("Topics learned by the best LDA model:")
topics = best_model.print_topics()
for topic in topics:
    print(topic)

with open('/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/topics.txt', 'w') as f:
    for topic in topics:
        f.write(str(topic) + '\n')

with open('/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/topics_per_document.txt', 'w') as f:
    for idx, doc in enumerate(documents):
        doc_bow = dictionary.doc2bow(doc)
        doc_topics = best_model.get_document_topics(doc_bow)  # Use best_model
        f.write(f"Document {idx}: {doc_topics}\n")

# Compute u_mass coherence score for best model
coherence_model_umass = CoherenceModel(model=best_model, texts=documents, dictionary=dictionary, coherence='u_mass')
coherence_umass = coherence_model_umass.get_coherence()
print('Coherence Score (u_mass): ', coherence_umass)

# Compute c_v coherence score for best model
coherence_model_cv = CoherenceModel(model=best_model, texts=documents, dictionary=dictionary, coherence='c_v')
coherence_cv = coherence_model_cv.get_coherence()
print('Coherence Score (c_v): ', coherence_cv)

print("Script completed successfully.")


# In[35]:


# Replace with your actual path
save_path = "/Users/juanpablocasado/Documents/Digital text/Digital Text (f)/best_model"

# Save the best model
best_model.save(save_path)


# In[22]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(best_model, corpus, dictionary)
vis


# In[36]:


# Extracting dominant topic and representative text for each document
results = []

for idx, doc in enumerate(documents):
    doc_bow = dictionary.doc2bow(doc)
    doc_topics = best_model.get_document_topics(doc_bow)
    dominant_topic = max(doc_topics, key=lambda x: x[1])
    
    # Extracting a snippet (first 10 words) from the document
    snippet = " ".join(doc[:10])
    
    results.append({
        "Document Index": idx,
        "Dominant Topic": dominant_topic[0],
        "Topic Probability": dominant_topic[1],
        "Representative Text": snippet
    })

# Creating a table (DataFrame) to display the results
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)

# Optionally, save the table to a CSV file for further analysis
results_df.to_csv("/Users/juanpablocasado/Downloads/document_topics.csv", index=False)


# In[37]:


# Adjust display settings
pd.set_option('display.max_rows', None)

# Display the DataFrame
results_df


# In[38]:


# Function to get sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to each document
results_df['Sentiment'] = results_df['Representative Text'].apply(get_sentiment)

# Categorize sentiment as Positive, Neutral, or Negative
def categorize_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

results_df['Sentiment Category'] = results_df['Sentiment'].apply(categorize_sentiment)

# Adjust display settings
pd.set_option('display.max_rows', None)

# Display the DataFrame
results_df


# In[39]:


# Set the visual style for the plot
sns.set_style("whitegrid")

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.countplot(x='Sentiment Category', data=results_df, order=['Positive', 'Neutral', 'Negative'], palette='viridis')

# Set the title and labels
plt.title('Distribution of Sentiment Categories', fontsize=16)
plt.xlabel('Sentiment Category', fontsize=14)
plt.ylabel('Number of Documents', fontsize=14)

# Display the plot
plt.show()


# In[ ]:




