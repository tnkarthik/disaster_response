import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """Function to read and merge data in csv files.

    Parameters
    ----------
    messages_filepath : str
        Path to csv file where messages data is stored.
    categories_filepath : str
        Path to csv file where categories data is stored.

    Returns
    -------
    DataFrame
        Merged data frame.

    """

    try:
        df_messages = pd.read_csv(messages_filepath)
        df_cats = pd.read_csv(categories_filepath)
        df = df_messages.merge(df_cats, on = "id", suffixes = ("", "_cats"), how = "right")
        return df
    except Exception as e:
        print("Failed to load data with error {0}".format(e))
        return pd.DataFrame()


def clean_data(df):
    """Function to clean categories variable .

    Parameters
    ----------
    df : DataFrame
        Input merged DataFrame.

    Returns
    -------
    DataFrame
        Output DataFrame with categories column cleaned and one hot encoded

    """

    #### Get categories colnames
    categories = df['categories']
    colnames = [x.split("-")[0] for x in categories.iloc[0].split(";")]

    #### Expand categories into individual features with numerical one-hot encodings
    categories = categories.str.split(";", expand = True).applymap(lambda x: int(x.split("-")[-1]))

    #### Update categories colnames
    categories.columns = colnames

    #### Modify original dataframe with categories columns
    df = df.drop(columns = ['categories'])
    df_out = pd.concat([df, categories], axis = 1)
    df_out = df_out.drop_duplicates()

    #### Additional transformation to change some of the values in the related
    #### category that are coded as 2 to 1. There are 193 such entries.
    #### I am assuming the related category means if there are any related
    #### previous messages or if the message is a new message that needs to be
    #### acted upon. So coding anything >1 as 1 makes sense in this case.

    df_out['related'] = df_out['related'].apply(lambda x: 1 if x > 0 else 0)

    return df_out



def save_data(df, database_filename):
    """Function to save cleaned data into database.

    Parameters
    ----------
    df : DataFrame
        Cleaned DataFrame with messages and categories data.
    database_filename : str
        Database name to save the table.

    Returns
    -------
    None
        None.

    """
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql('disaster_response', engine, index = False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
