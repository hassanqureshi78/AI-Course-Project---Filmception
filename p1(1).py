#!/usr/bin/env python3
"""
Movie Data Preprocessor for CSV Generation
"""

import pandas as pd
import ast
import nltk
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Constants
MIN_GENRE_SAMPLES = 5    # Minimum movies per genre
MAX_GENRE_SAMPLES = 1000 # Maximum movies per genre
TEST_SIZE = 0.2          # Test set proportion

def clean_text(text):
    """Enhanced text cleaning pipeline"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # Contraction handling
    contractions = {
        r"won't": "will not", r"can't": "cannot", r"n't": " not",
        r"'re": " are", r"'s": " is", r"'d": " would", 
        r"'ll": " will", r"'t": " not", r"'ve": " have", 
        r"'m": " am", r"what's": "what is"
    }
    
    for pat, repl in contractions.items():
        text = re.sub(pat, repl, text)
    
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", ' ', text)
    text = text.lower()
    
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if (token not in stop_words) and (len(token) > 2) and (token.isalpha())
    ]
    
    return ' '.join(tokens)

def load_metadata(metadata_path):
    """Load and validate genre metadata"""
    try:
        df = pd.read_csv(
            metadata_path, 
            sep='\t', 
            header=None,
            usecols=[0, 8], 
            names=['movie_id', 'genres'],
            dtype={'movie_id': str}
        )
        df.dropna(inplace=True)
        
        def parse_genres(genre_json):
            try:
                genres_dict = ast.literal_eval(genre_json)
                return [
                    re.sub(r'[^a-z]', '', g.lower().strip())
                    for g in genres_dict.values()
                    if len(g.strip()) >= 4
                ]
            except:
                return []
        
        df['genres'] = df['genres'].apply(parse_genres)
        return df[df['genres'].map(len) > 0]
    
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        raise

def load_summaries(summaries_path):
    """Load plot summaries"""
    try:
        return pd.read_csv(
            summaries_path,
            sep='\t',
            header=None,
            names=['movie_id', 'summary'],
            dtype={'movie_id': str}
        ).dropna()
    except Exception as e:
        print(f"❌ Error loading summaries: {e}")
        raise

def balance_genres(df):
    """Balance genre distribution with proper scoping"""
    exploded = df.explode('genres')
    genre_counts = exploded['genres'].value_counts()
    
    # Define valid_genres here so it's accessible later
    global valid_genres
    valid_genres = genre_counts[genre_counts >= MIN_GENRE_SAMPLES].index.tolist()
    
    if not valid_genres:
        raise ValueError("No genres meet the minimum sample threshold")
    
    df = df[df['genres'].apply(lambda x: any(g in valid_genres for g in x))]
    
    balanced_dfs = []
    for genre in valid_genres:
        genre_movies = df[df['genres'].apply(lambda x: genre in x)]
        if len(genre_movies) > MAX_GENRE_SAMPLES:
            genre_movies = genre_movies.sample(MAX_GENRE_SAMPLES, random_state=42)
        balanced_dfs.append(genre_movies)
    
    balanced_df = pd.concat(balanced_dfs).drop_duplicates(subset=['movie_id'])
    
    # Filter rare genre combinations
    genre_combos = balanced_df['genres'].apply(lambda x: tuple(sorted(x)))
    combo_counts = genre_combos.value_counts()
    valid_combos = combo_counts[combo_counts >= 2].index
    balanced_df = balanced_df[balanced_df['genres'].apply(
        lambda x: tuple(sorted(x)) in valid_combos
    )]
    
    print("\n🎭 Final Genre Distribution:")
    print(balanced_df.explode('genres')['genres'].value_counts())
    
    return balanced_df

def get_genre_tuple(genres):
    """Stratification helper using the global valid_genres"""
    return tuple(sorted(set(g for g in genres if g in valid_genres)))

def preprocess_data(metadata_path, summaries_path):
    print("🚀 Starting data preprocessing pipeline...")
    
    try:
        # Load and merge data
        meta = load_metadata(metadata_path)
        summaries = load_summaries(summaries_path)
        df = pd.merge(summaries, meta, on='movie_id')
        print(f"Initial dataset size: {len(df)} movies")
        
        # Clean text
        tqdm.pandas(desc="Cleaning summaries")
        df['cleaned_summary'] = df['summary'].progress_apply(clean_text)
        
        # Balance genres
        balanced_df = balance_genres(df)
        print(f"Balanced dataset size: {len(balanced_df)} movies")
        
        # Train-test split
        train, test = train_test_split(
            balanced_df[['movie_id', 'cleaned_summary', 'genres']],
            test_size=TEST_SIZE,
            random_state=42,
            stratify=balanced_df['genres'].apply(get_genre_tuple)
        )
        
        # Save outputs to Dataset folder
        train.to_csv("Dataset/train_cleaned.csv", index=False)
        test.to_csv("Dataset/test_cleaned.csv", index=False)
        print("\n✅ Successfully created:")
        print(f"- Dataset/train_cleaned.csv ({len(train)} movies)")
        print(f"- Dataset/test_cleaned.csv ({len(test)} movies)")
        
        return train, test
        
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data('Dataset/movie.metadata.csv', 'Dataset/plot_summaries.txt')