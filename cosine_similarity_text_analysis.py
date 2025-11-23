import nltk
nltk.download()
from nltk.book import *
import nltk.tokenize
import numpy as np
import pandas as pd
import string
pd.set_option('max_columns', None)
from collections import Counter
import math
nltk.download('stopwords', quiet=True)
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = nltk.corpus.stopwords.words('english')
sentence1 = "Thomas Jefferson began building Monticello at the age of 26."              #Ορίζουμε τις προτάσεις
sentence2 = "Matrices, vector spaces, and information retrieval."
token1=sentence1.split()                                                                #Παράγουμε την tokenized λίστα με την split
token2=sentence2.split()
tokenn1=nltk.word_tokenize(sentence1)                                                   #Παράγουμε την tokenized λίστα με την βοήθεια του nltk
tokenn2=nltk.word_tokenize(sentence2)
vocab1 = sorted(set(token1))                                                            #Ταξινομούμε έτσι ώστε να προηγούνται τα νούμερα και μετά τα κεφαλάια
vocab2 = sorted(set(token2))
vocabn1 = sorted(set(tokenn1))
vocabn2 = sorted(set(tokenn2))
print(', '.join(vocab1))                                                                #Τα tokens χωρίζουμε με κόμμα και κενό
print(', '.join(vocab2))
print(', '.join(vocabn1))
print(', '.join(vocabn2))
num_tokens1 = len(token1)                                                               #Βρίσκουμε το μέγεθος κάθε tokenized λίστας       
num_tokens2 = len(token2)
num_tokensn1 = len(tokenn1)
num_tokensn2 = len(tokenn2)
vocab_size1 = len(vocab1)                                                               #Βρίσκουμε το μέγεθος κάθε αλφαβήτου
vocab_size2 = len(vocab2)
vocab_sizen1 = len(vocabn1)
vocab_sizen2 = len(vocabn2)
onehot_vectors1 = np.zeros((num_tokens1, vocab_size1), int)                             #Φτιάχνουμε τα vector και τα γεμίζουμε με 0 σε όλο το μέγεθος του αλφαβήτου
onehot_vectors2 = np.zeros((num_tokens2, vocab_size2), int)
onehot_vectorsn1 = np.zeros((num_tokensn1, vocab_sizen1), int)
onehot_vectorsn2 = np.zeros((num_tokensn2, vocab_sizen2), int)
for i, word in enumerate(token1):
    onehot_vectors1[i, vocab1.index(word)] = 1                                          #Όπου έχουμε ταίριασμα για κάθε λέξη με token της αντίστοιχης λίστα βάζουμε 1 στον vector
for i, word in enumerate(token2):
    onehot_vectors2[i, vocab2.index(word)] = 1
for i, word in enumerate(tokenn1):
    onehot_vectorsn1[i, vocabn1.index(word)] = 1
for i, word in enumerate(tokenn2):
    onehot_vectorsn2[i, vocabn2.index(word)] = 1
print(onehot_vectors1)                                                                  #Τυπώνουμε τα vectors    
print(onehot_vectors2)
print(onehot_vectorsn1)
print(onehot_vectorsn2)
print(pd.DataFrame(onehot_vectors1, columns=vocab1))                                    #Τυπώνουμε τα vector με τη χρήση του pandas και βλέπουμε την διαφορά
print(pd.DataFrame(onehot_vectors2, columns=vocab2))
print(pd.DataFrame(onehot_vectorsn1, columns=vocabn1))
print(pd.DataFrame(onehot_vectorsn2, columns=vocabn2))

sentence3="Auto-encoding of documents for information retrieval systems."               #Ορίζουμε τις επιπλέον προτάσεις
sentence4="Their government gave these documents publicity."
sentence5="It was a government business agency."
corpus = {}                                                                             #Κατασκευάζουμε το dictionary
corpus['sent2'] = dict((token.strip('.'), 1) for token in sentence2.split())            #Το γεμίζουμε για κάθε πρόσταση, βγάζοντας τις τελείες και αφού πρώτα κάνουμε tokenization με την split
corpus['sent3'] = dict((token.strip('.'), 1) for token in sentence3.split())
corpus['sent4'] = dict((token.strip('.'), 1) for token in sentence4.split())
corpus['sent5'] = dict((token.strip('.'), 1) for token in sentence5.split())
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T                          #Φτιάχνουμε το dataframe και το τυπώνουμε στην από κάτω γραμμή
print(df)
dfs=df.T
print("similarity of sent2 and sent3:", dfs.sent2.dot(dfs.sent3))                       #Συγκρίνουμε τις ομοιότητες κάθε πρότασης με τις υπόλοιπες
print("similarity of sent2 and sent4:", dfs.sent2.dot(dfs.sent4))
print("similarity of sent2 and sent5:", dfs.sent2.dot(dfs.sent5))
print("similarity of sent3 and sent4:", dfs.sent3.dot(dfs.sent4))
print("similarity of sent3 and sent5:", dfs.sent3.dot(dfs.sent5))
print("similarity of sent4 and sent5:", dfs.sent4.dot(dfs.sent5))
corpusb = {}                                                                            #Κατασκευάζουμε το dictionary για τις πρώτες 50 λέξεις του text4 και text7
corpusb['text4'] = dict((token, 1) for token in text4[:50])
corpusb['text7'] = dict((token, 1) for token in text7[:50])
dfb = pd.DataFrame.from_records(corpusb).fillna(0).astype(int).T                        #Φτιάχνουμε το dataframe και το τυπώνουμε στην από κάτω γραμμή
print(dfb)
dfsb=dfb.T
print("similarity of text4 and text7:", dfsb.text4[:50].dot(dfsb.text7[:50]))           #Συγκρίνουμε την ομοιότητα
print([(k, v) for (k, v) in (dfsb.text4 & dfsb.text7).items() if v])                    #Εμφανίζουμε τις κοινές λέξεις

table4 = str.maketrans('', '', '\t')                                                    #Αφαιρούμε τα σημεία στίξης για τις πρώτες 50 λέξεις του text4
token_list4 = [word.translate(table4) for word in text4[:50]]
punctuations4 = (string.punctuation).replace("'", "")
trans_table4 = str.maketrans('', '', punctuations4)
stripped_words4 = [word.translate(trans_table4) for word in token_list4]
token_list4 = [str for str in stripped_words4 if str]
token_list4 =[word.lower() for word in token_list4]                                     #Μετατρέπουμε κάθε λέξη σε πεζά γράμματα για τις πρώτες 50 λέξεις του text4
token_list4 = [x for x in token_list4 if x not in stopwords]                            #Αφαιρούμε τα προθήματα για τις πρώτες 50 λέξεις του text4
table7 = str.maketrans('', '', '\t')                                                    #Αφαιρούμε τα σημεία στίξης για τις πρώτες 50 λέξεις του text7
token_list7 = [word.translate(table7) for word in text7[:50]]
punctuations7 = (string.punctuation).replace("'", "")
trans_table7 = str.maketrans('', '', punctuations7)
stripped_words7 = [word.translate(trans_table7) for word in token_list7]
token_list7 = [str for str in stripped_words7 if str]
token_list7 =[word.lower() for word in token_list7]                                     #Μετατρέπουμε κάθε λέξη σε πεζά γράμματα για τις πρώτες 50 λέξεις του text7
token_list7 = [x for x in token_list7 if x not in stopwords]                            #Αφαιρούμε τα προθήματα για τις πρώτες 50 λέξεις του text7
pos_index4 = {}                                                                         #Δημιουργούμε τα dictionaries
pos_index7 = {}
for pos, term in enumerate(token_list4):                                                #Για κάθε όρο και θέση στην κανονικοποιημένη λίστα για τις πρώτες 50 λέξεις του text4
    if term in pos_index4:                                                              #Αν το token βρίσκεται στη κανονικοποιημένη λίστα για τις πρώτες 50 λέξεις του text4
        pos_index4[term][0] = pos_index4[term][0] + 1                                   #Αυξάνουμε κατά 1 το πλήθος του για τις πρώτες 50 λέξεις του text4
        if 4 in pos_index4[term][1]:                                                    #Αν το έγγραφο 4 έχει τον όρο για τις πρώτες 50 λέξεις του text4
            pos_index4[term][1][4].append(pos)                                          #Συμπληρώνουμε στο τέλος τη θέση του για τις πρώτες 50 λέξεις του text4
    else:                                                                               #Αν το token δεν περιέχεται στην κανονικοποιημένη λίστα δεν το πρόσθέτουμε και συνεχίζουμε για το επόμενο για τις πρώτες 50 λέξεις του text4
        pos_index4[term] = []
        pos_index4[term].append(1)
        pos_index4[term].append({})
        pos_index4[term][1][4] = [pos]
for pos, term in enumerate(token_list7):                                                #Για κάθε όρο και θέση στην κανονικοποιημένη λίστα για τις πρώτες 50 λέξεις του text7    
    if term in pos_index7:                                                              #Αν το token βρίσκεται στη κανονικοποιημένη λίστα για τις πρώτες 50 λέξεις του text7
        pos_index7[term][0] = pos_index7[term][0] + 1                                   #Αυξάνουμε κατά 1 το πλήθος του για τις πρώτες 50 λέξεις του text7
        if 7 in pos_index7[term][1]:                                                    #Αν το έγγραφο 4 έχει τον όρο για τις πρώτες 50 λέξεις του text7
            pos_index7[term][1][7].append(pos)                                          #Συμπληρώνουμε στο τέλος τη θέση του για τις πρώτες 50 λέξεις του text7
    else:                                                                               #Αν το token δεν περιέχεται στην κανονικοποιημένη λίστα δεν το πρόσθέτουμε και συνεχίζουμε για το επόμενο για τις πρώτες 50 λέξεις του text7
        pos_index7[term] = []
        pos_index7[term].append(1)
        pos_index7[term].append({})     
        pos_index7[term][1][7] = [pos]
fdist4 = FreqDist(token_list4[:50])                                                     #Βρίσκουμε τη συχνότητα κάθε όρου στο κείμενο 4 για τις πρώτες 50 λέξεις
c4=fdist4.most_common(3)                                                                #Βρίσκουμε τις 3 λέξεις με τη μεγαλύτερη συχνότητα στο κείμενο 4 για τις πρώτες 50 λέξεις
print("three most common words in first 50 words in text4:")
print(c4)                                                                               #Τις τυπώνουμε
pos_idx14 = pos_index4[c4[0][0]]                                                        
pos_idx24 = pos_index4[c4[1][0]]                                                        
pos_idx34 = pos_index4[c4[2][0]]                                                        
print("Positional Index of text 4 fist 50 words, 3 most common words:")
print(pos_idx14)                                                                        #Τυπώνουμε τα posting lists τους (ΠΡΩΤΗ)    
print(pos_idx24)                                                                        #Τυπώνουμε τα posting lists τους (ΔΕΥΤΕΡΗ)
print(pos_idx34)                                                                        #Τυπώνουμε τα posting lists τους (ΤΡΙΤΗ)
fdist7 = FreqDist(token_list7[:50])                                                     #Βρίσκουμε τη συχνότητα κάθε όρου στο κείμενο 7 για τις πρώτες 50 λέξεις
c7=fdist7.most_common(3)                                                                #Βρίσκουμε τις 3 λέξεις με τη μεγαλύτερη συχνότητα στο κείμενο 7 για τις πρώτες 50 λέξεις
print("three most common words in first 50 words in text7:")
print(c7)                                                                               #Τις τυπώνουμε
pos_idx17 = pos_index7[c7[0][0]]
pos_idx27 = pos_index7[c7[1][0]]
pos_idx37 = pos_index7[c7[2][0]]
print("Positional Index of text 7 fist 50 words, 3 most common words:")
print(pos_idx17)                                                                        #Τυπώνουμε τα posting lists τους (ΠΡΩΤΗ)
print(pos_idx27)                                                                        #Τυπώνουμε τα posting lists τους (ΔΕΥΤΕΡΗ)        
print(pos_idx37)                                                                        #Τυπώνουμε τα posting lists τους (ΤΡΙΤΗ)

table2 = str.maketrans('', '', '\t')                                                    #Αφαιρούμε τα σημεία στίξης για την πρόταση 2 tokenized με το nltk
token_list2 = [word.translate(table2) for word in tokenn2]
punctuations2 = (string.punctuation).replace("'", "")
trans_table2 = str.maketrans('', '', punctuations2)
stripped_words2 = [word.translate(trans_table2) for word in token_list2]
token_list2 = [str for str in stripped_words2 if str]
token_list2 =[word.lower() for word in token_list2]                                     #Μετατρέπουμε κάθε λέξη σε πεζά γράμματα για την πρόταση 2
token_list2 = [x for x in token_list2 if x not in stopwords]                            #Αφαιρούμε τα προθήματα για την πρόταση 2
bag_of_words2 = Counter(token_list2)                                                    #Δημιουργούμε το bag_of_words για την πρόταση 2
print("Bag of words for normalized sent2:")
print(bag_of_words2)                                                                    #Το τυπώνουμε
bag_of_words4 = Counter(token_list4)                                                    #Δημιουργούμε το bag_of_words για τις πρώτες 50 λέξεις του κειμένου 4
print("Bag of words for first 50 word from text4:")
print(bag_of_words4)                                                                    #Το τυπώνουμε
bag_of_words7 = Counter(token_list7)                                                    #Δημιουργούμε το bag_of_words για τις πρώτες 50 λέξεις του κειμένου 7
print("Bag of words for first 50 word from text7:")
print(bag_of_words7)                                                                    #Το τυπώνουμε
text4vector = []                                                                        #Φτιάνουμε το vector για τις πρώτες 50 λέξεις του κειμένου 4
text7vector = []                                                                        #Φτιάνουμε το vector για τις πρώτες 50 λέξεις του κειμένου 7
text4length = len(token_list4)                                                          #Βρίσκουμε το μέγεθος της λίστας για τις πρώτες 50 λέξεις του κειμένου 4
text7length = len(token_list7)                                                          #Βρίσκουμε το μέγεθος της λίστας για τις πρώτες 50 λέξεις του κειμένου 7
for key, value in bag_of_words4.most_common():                                          #Γεμίζουμε τον vector για τις πρώτες 50 λέξεις του κειμένου 4 αναλόγως με το μέγεθός του  
    text4vector.append(value / text4length)
for key, value in bag_of_words7.most_common():                                          #Γεμίζουμε τον vector για τις πρώτες 50 λέξεις του κειμένου 7 αναλόγως με το μέγεθός του  
    text7vector.append(value / text7length)
#print(text4vector)
#print(text7vector)
def cosine_sim(vec1, vec2):                                                             #Φτιάχνουμε την συνάρτηση που δέχεται 2 vectors και μέσα από μαθηματικές πράξεις, υπολογίζει την συνιμιτονοειδής ομοιότητά τους
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    return dot_prod / (mag_1 * mag_2)
sim=cosine_sim(text4vector, text7vector)                                                #Καλούμε την συνάρτηση για τα 2 vectors
print("Cosine similarity of first 50 word of books 4 and 7:")
print(sim)                                                                              #Τυπώνουμε την ομοιότητά τους
table4f = str.maketrans('', '', '\t')
token_list4f = [word.translate(table4f) for word in text4]                              #Αφαιρούμε τα σημεία στίξης για ολόκληρο το κείμενο 4
punctuations4f = (string.punctuation).replace("'", "")
trans_table4f = str.maketrans('', '', punctuations4f)
stripped_words4f = [word.translate(trans_table4f) for word in token_list4f]
token_list4f = [str for str in stripped_words4f if str]
token_list4f =[word.lower() for word in token_list4f]                                   #Μετατρέπουμε ολόκληρο το κείμενο 4 σε πεζά γράμματα
token_list4f = [x for x in token_list4f if x not in stopwords]                          #Αφαιρούμε τα προθήματα από ολόκληρο το κείμενο 4    
table7f = str.maketrans('', '', '\t')                                                   #Αφαιρούμε τα σημεία στίξης για ολόκληρο το κείμενο 7
token_list7f = [word.translate(table7f) for word in text7]
punctuations7f = (string.punctuation).replace("'", "")
trans_table7f = str.maketrans('', '', punctuations7f)
stripped_words7f = [word.translate(trans_table7f) for word in token_list7f]
token_list7f = [str for str in stripped_words7f if str]
token_list7f =[word.lower() for word in token_list7f]                                   #Μετατρέπουμε ολόκληρο το κείμενο 7 σε πεζά γράμματα
token_list7f = [x for x in token_list7f if x not in stopwords]                          #Αφαιρούμε τα προθήματα από ολόκληρο το κείμενο 7
bag_of_words4f = Counter(token_list4f)                                                  #Δημιουργούμε το bag_of_words για ολόκληρο το κείμενο 4
#print(bag_of_words4f)
bag_of_words7f = Counter(token_list7f)                                                  #Δημιουργούμε το bag_of_words για ολόκληρο το κείμενο 7
#print(bag_of_words7f)
text4fvector = []                                                                       #Αρχικοποιούμε το vector για ολόκληρο το κείμενο 4
text7fvector = []                                                                       #Αρχικοποιούμε το vector για ολόκληρο το κείμενο 7
text4flength = len(token_list4f)                                                        #Βρίσκουμε το μέγεθος για ολόκληρο το κείμενο 4
text7flength = len(token_list7f)                                                        #Βρίσκουμε το μέγεθος για ολόκληρο το κείμενο 7
for key, value in bag_of_words4f.most_common():
    text4fvector.append(value / text4flength)                                           #Γεμίζουμε τον vector για ολόκληρο το κείμενο 4 αναλόγως με το μέγεθός του        
for key, value in bag_of_words7f.most_common():
    text7fvector.append(value / text7flength)                                           #Γεμίζουμε τον vector για ολόκληρο το κείμενο 7 αναλόγως με το μέγεθός του  
#print(text4fvector)
#print(text7fvector)
simf=cosine_sim(text4fvector, text7fvector)                                             #Καλούμε την συνάρτηση που υπολογίζει την συνιμιτονοειδής ομοιότητά των 2 κειμένων
print("Cosine similarity of full books 4 and 7:")
print(simf)                                                                             #Την εμφανίζουμε

# docs=sent4+sent7
# corpuss=docs
# vectorizer = TfidfVectorizer(min_df=1)                                                 #Δημιουργούμε το vector για την πρόταση 4 και την πρόταση 7 για να υπολογίσουμε την tf-idf ομοιότητα
# models = vectorizer.fit_transform(corpuss)
# print(models.todense().round(2))                                                       #Εμφανίζουμε το vector
#simsent=cosine_sim(model4, model7)
#print("Cosine similarity of sent4 sent7:")
#print(simsent)