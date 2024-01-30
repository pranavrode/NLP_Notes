import contractions
from unidecode import unidecode
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# remove newlines 
def remove_lines(data):
    clean_text =  data.replace("\n",' ').replace("\\n",' ').replace("\t",' ')
    return clean_text

# contraction mapping 
def expand_text(data):
    expanded_doc = contractions.fix(data)
    return expanded_doc

# handle accented character

def accented_char(data):
    fixed_text = unidecode(data)
    return fixed_text

# clean data \
stopword_list = stopwords.words("english")
stopword_list.remove("not")
stopword_list.remove("no")
stopword_list.remove("nor")
def clean_data(data):
    tokens = word_tokenize(data)
    normalization = [word.lower() for word in tokens]
    remove_punct = [word for word in normalization if word not in punctuation]
    words_without_stop = [word for word in  remove_punct if word not in stopword_list]
    clean_text = [word for word in words_without_stop if len(word)>2]
    return clean_text

# lemmatization
def lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    final_text = []
    for word in data:
        lemmatized_word = lemmatizer.lemmatize(word)
        final_text.append(lemmatized_word)
    return final_text

def join_list(data):
    return " ".join(data)