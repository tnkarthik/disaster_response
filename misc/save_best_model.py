import sys
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import joblib



def tokenize(text):
    """Customer tokenizer using nlkt functions.

    Parameters
    ----------
    text : str
        Input text to be tokenized.

    Returns
    -------
    str
        Tokenized text.

    """

    #### Normalize (convert to lower case and remove punctuation) text
    text = re.sub("[^a-z,A-Z,0-9]", " ", text.lower().strip())

    #### Tokenize text to words
    text = word_tokenize(text)

    #### Remove stop words
    text = [i for i in text if i not in stopwords.words('english') ]

    #### Lemmatize
    text = [WordNetLemmatizer().lemmatize(x, pos = 'n') for x in text]
    text = [WordNetLemmatizer().lemmatize(x, pos = 'v') for x in text]

    return text

model = joblib.load("../models/optimized_rf_classifier.pkl")
joblib.dump(model.best_estimator_, '../models/rf_classifier.pkl', compress = 1)
