# Project Documentation

## 1. imports and setup
In this section, the script prepares the environment for text processing and analysis. Here's a detailed breakdown:

### Modules and libraries
- **os:** This module provides a way of interacting with the operating system, such as reading and writing files.
- **re:** This module allows for regular expression operations, which is useful for text cleaning and manipulation.
- **nltk:** The Natural Language Toolkit is a set of libraries and tools for symbolic and statistical natural language processing (NLP).
- **matplotlib.pyplot and seaborn:** These are visualization libraries that help in plotting and visualizing data.
- **tqdm:** This module provides a progress bar, useful for tracking the progress of loops.
- **logging:** This module is used for logging messages.

### Nltk resources
- **stopwords:** Common words like 'and', 'the', 'is', etc., which are often removed in text processing.
- **wordnet:** A lexical database for the English language, useful for lemmatization.
- **averaged_perceptron_tagger:** A tagger for part-of-speech tagging.

## 2. directory definitions
In this section, the script specifies the directories where the text documents are located and where the processed documents will be saved.

Directories
- **source_dir:** This is the directory where all the original documents in TXT format are located. The script will read documents from this directory for processing.
- **output_dir:** This is the directory where all the cleaned and processed documents in TXT format will be saved. The script will save the results of its cleaning operations here.

By defining these directories, the script sets the stage for batch processing of documents, ensuring efficient and organized processing and storage of results.

## 3. text cleaning functions
Text cleaning is a crucial step in any text processing or analysis task. In this section, two functions are defined to perform text cleaning:

### Clean_text_without_lemmatization(text)
This function performs the following steps:

- Converts the text to lowercase.
- Removes any character that is not an alphabet.
- Tokenizes the text into words.
- Removes stopwords (common words like 'and', 'the', 'is', etc.).
- Returns the cleaned words.

### Clean_text_with_lemmatization(text)
This function uses the previous function for initial cleaning and then performs lemmatization. Lemmatization is the process of converting words to their base or root form. For example, "running" becomes "run", and "better" becomes "good". The steps are:

- Use clean_text_without_lemmatization(text) for initial cleaning.
- Use the WordNetLemmatizer to lemmatize words based on their part-of-speech tags.
- Return the lemmatized words.

These functions ensure that the text is in a clean and standardized format, suitable for further processing and analysis.

## 4. processing the corpus
In this section, the script reads documents from the source directory, cleans them, and then saves the cleaned documents to the output directory.

Steps:
### A. reading documents:
The script lists all files in the source directory and reads them one by one.
### B. cleaning:
For each document, the script uses the clean_text_with_lemmatization function to clean and lemmatize the text.
### C. saving cleaned documents:
After cleaning, the processed text is saved to the output directory with the same filename as the original. This ensures that original files remain untouched, and cleaned versions are stored separately.

This section ensures that all documents are ready for further analysis by converting them into a clean and standardized format.

### 5. creating a dictionary and corpus for topic modeling
Topic modeling requires two main structures: a dictionary and a corpus.

Steps:
### A. building a dictionary:
A dictionary maps every word to a unique ID. This script uses the Gensim library to build a dictionary from the cleaned documents.
### B. building a corpus:
A corpus is a collection of documents represented in a format suitable for topic modeling. In this script, the corpus is represented as a bag-of-words (BoW) format, where each document is a list of (word ID, word frequency) pairs.
### C. applying term frequency-inverse document frequency (tf-idf):
TF-IDF is a way to weight terms based on how frequent they are in a document versus how frequent they are across all documents. The script applies this weighting to the BoW corpus to create a TF-IDF corpus.

By the end of this section, the script has created structures that are ready to be used for topic modeling.

## 6. topic modeling

Topic modeling is a type of statistical model used for discovering abstract topics in a collection of documents. In this script, Latent Dirichlet Allocation (LDA) is employed as the topic modeling technique.

Steps:
### A. building the lda model:
Using the previously created dictionary and corpus, the script builds an LDA model. This model will try to discover a specified number of topics from the corpus.
### B. displaying topics:
Once the model is built, it displays the topics discovered along with the top words associated with each topic. These words give an insight into what each topic might represent.
### C. visualizing topics:
The script uses pyLDAvis, a Python library, to visualize the topics. This visualization provides an interactive way to explore the topics, see their prevalence, and understand the words associated with each topic.

This section enables the user to understand the major themes or topics that are present in the corpus of documents.

## 7. assigning dominant topics to documents

After discovering the topics in the corpus, it's helpful to know which topic is most prominent in each document.

Steps:
### A. getting dominant topic for each document:
For each document in the corpus, the script determines the topic that has the highest probability of representing that document. This is termed the dominant topic.
### B. extracting a snippet from each document:
To give context, the script extracts the first few words from each document as a representative snippet.
### C. storing results in a dataframe:
The document index, dominant topic, topic probability, and representative text snippet are stored in a DataFrame. This tabular format makes it easier to analyze and visualize the results.
### D. saving results to csv:
Optionally, the results can be saved to a CSV file for further analysis or sharing.

This section provides a clear mapping between each document and its dominant topic, allowing for an organized overview of the corpus.

## 8. sentiment analysis

Sentiment analysis is the process of determining the emotional tone or sentiment behind a piece of text. In this section, the script calculates the sentiment for each document and categorizes it as Positive, Neutral, or Negative.

Steps:
### Calculating sentiment score:
The script uses the TextBlob library to compute a sentiment score for each document's representative snippet. The score ranges between -1 (most negative) to 1 (most positive).
### Categorizing sentiment:
Based on the sentiment score, each document is categorized as:

  - Positive if the score is greater than 0.
  - Neutral if the score is 0.
  - Negative if the score is less than 0.
    
### Storing results in a dataframe:
The sentiment scores and categories are added to the previously created DataFrame, allowing for a consolidated view of the topic and sentiment for each document.

This section offers insights into the emotional tone of the documents, which can be crucial for understanding the context and nuances of the content.

## 9. visualization
Visualization provides an intuitive way to understand and interpret the results. In this section, the script visualizes the distribution of sentiment categories across the documents.

Steps:
### A. setting visualization style:
The script uses the seaborn library to set a visual style for the plot.
### B. creating a bar plot:
A bar plot is created to display the count of documents for each sentiment category (Positive, Neutral, Negative).
### C. adding titles and labels:
Relevant titles and labels are added to the plot to provide context and clarity.

This visualization offers a quick snapshot of the overall sentiment distribution in the corpus, making it easier to gauge the general tone of the documents.



