import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Description:
    Load and merge messages and categories datasets
    
    Arguments:
    messages_filepath: string. Path to message data file
    categories_filepath: string. Path to categories data file
       
    Returns:
    df: dataframe. Dataframe that loaded with messages and categories data
    """
    
    # Load data from paths
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories, how = 'left', on = ['id'])


def clean_data(df):
    """
    Description:
    This method removes duplicate data and converts categories to numerical value
    
    Arguments:
    df: dataframe. The dataframe with original data from csv
       
    Returns:
    df: dataframe. Cleaned data
    """
    
    # Create columns for each category
    categories = df['categories'].str.split(';', expand = True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert  category values to numeric values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the `categories` as we don't need it anymore
    df = df.drop('categories', axis = 1)
    
    
    # Concatenate the new columns to existing dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace = True)
    
    # Remove rows with unwanted value
    df = df[df['related'] != 2]
    
    return df


def save_data(df, database_filename):
    """
    Description:
    Store data to SQLite
    
    Arguments:
    df: dataframe. A cleaned dataframe.
    database_filename: string. SQLite file.
       
    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


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