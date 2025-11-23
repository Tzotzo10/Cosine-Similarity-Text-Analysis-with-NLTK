Cosine Similarity Text Analysis with NLTK
Overview

This Python project analyzes text similarity using tokenization, normalization, bag-of-words, and cosine similarity. It demonstrates:

One-hot encoding of sentences

Bag-of-words representation

Tokenization (with split and nltk.word_tokenize)

Normalization (lowercasing, removing punctuation, and stopwords)

Cosine similarity between text vectors

Frequency distribution analysis of words

The code works with small example sentences and excerpts from texts (text4 and text7 from NLTK’s book corpus).

Requirements

Python 3.8+

Libraries:

nltk

numpy

pandas

scipy

sklearn (optional, commented out)

Install dependencies using pip:

pip install nltk numpy pandas scipy scikit-learn


Additionally, download NLTK resources (if not already):

import nltk
nltk.download('book')
nltk.download('punkt')
nltk.download('stopwords')

Usage

Run the script:
The script performs the following steps:

Tokenizes sentences using split and nltk.word_tokenize

Generates vocabulary and one-hot vectors

Creates dictionaries for Bag-of-Words representation

Computes cosine similarity between example sentences and texts

Cleans text from punctuation and stopwords

Computes frequency distributions for analysis

View outputs:

Vocabulary lists

One-hot vectors (printed and via pandas DataFrames)

Bag-of-words counts

Cosine similarity scores for both first 50 words and full texts

Example Output
Vocabulary for sentence1: Thomas, Jefferson, began, building, Monticello, at, the, age, of, 26.
One-hot vector for sentence1:
[[0 0 1 0 ...]]
Bag-of-words for first 50 words of text4:
Counter({'the': 5, 'and': 3, ...})
Cosine similarity of first 50 words of books 4 and 7: 0.543
Cosine similarity of full books 4 and 7: 0.678

Code Structure

Tokenization: Split and NLTK word tokenizer

Normalization: Lowercasing, punctuation removal, stopword removal

Vectorization: One-hot vectors, bag-of-words vectors

Similarity Calculation: Cosine similarity function

Frequency Analysis: nltk.FreqDist

Notes

Stopwords are removed using NLTK’s English stopwords list.

Punctuation is removed using Python’s string.punctuation.

The code demonstrates both small sentence-level similarity and full-text similarity.

Optional TF-IDF analysis is commented out but can be used with sklearn.feature_extraction.text.TfidfVectorizer.

Author

Georgia Papapanagiotou