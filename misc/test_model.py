import sys
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import pandas as pd
import re

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    """Load data from the database.

    Parameters
    ----------
    database_filepath : str
        Database filepath.

    Returns
    -------
    tuple
        Tuple of features, targets and target names.

    """

    #### Establish connection to database and read the table as pandas df
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)

    #### define message (X), categories (Y) and category names
    X = df['message']
    df_y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    category_names = df_y.columns.tolist()
    Y = df_y.values

    return X, Y, category_names


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


def evaluate_model(model, X_test, Y_test, category_names):
    """Short summary.

    Parameters
    ----------
    model : type
        Description of parameter `model`.
    X_test : type
        Description of parameter `X_test`.
    Y_test : type
        Description of parameter `Y_test`.
    category_names : type
        Description of parameter `category_names`.

    Returns
    -------
    type
        Description of returned object.

    """
    Y_model = model.predict(X_test)
    print('Calculating F1-scores.......')

    try:

        f1_scores = f1_score(Y_test[:,1:], Y_model[:,1:], average = None)
        precision = precision_score(Y_test[:,1:], Y_model[:,1:], average = None)
        recall = recall_score(Y_test[:,1:], Y_model[:,1:], average = None)

        print("Category wise F1_score: {0}".format(list(zip(category_names[1:], f1_scores))))
        print("Category wise Precision: {0}".format(list(zip(category_names[1:], precision))))
        print("Category wise Recall: {0}".format(list(zip(category_names[1:], recall))))
        
    except Exception as e:
        print("Failed with exception {0}".format(e))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = joblib.load(model_filepath)

        print('Evaluating model...')
        evaluate_model(model, X, Y, category_names)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'test_model.py ../data/DisasterResponse.db ../models/optimized_rf_classifier.pkl')


if __name__ == '__main__':
    main()
