import requests
import json
import numpy as np
from newspaper import Article
import feedparser
import pandas as pd
import nltk 
import matplotlib
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake 
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

def get_news_links(query, num_results=5):
    query = query.replace(" ", "%20")  # URL-encode spaces
    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        print("Failed to retrieve RSS feed or no entries found.")
        return []

    links = []
    for entry in feed.entries[:num_results]:
        links.append(entry.link)
    
    return links

def get_article_details(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve article. Status code: {response.status_code}")
        return None
    
    article = Article(url)
    article.set_html(response.text)
    article.parse()
    
    return {
        'title': article.title,
        'author': article.authors,
        'publish_date': article.publish_date,
        'text': article.text,
        'top_image': article.top_image,
        'videos': article.movies,
        'keywords': article.keywords,
        'summary': article.summary,
        'url': url
    }

def preprocess_data():
    df = pd.read_csv('news_articles.csv')
    df = df.dropna()
    df['combined_text'] = df['title'].map(str) + " " + df['content'].map(str)
    return df

# Function to remove non-ascii characters from the text
def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)
# function to remove the punctuations, apostrophe, special characters using regular expressions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text


# stop words are the words that convery little to no information about the actual content like the words:the, of, for etc
def remove_stopwords(word_tokens):
    filtered_sentence = [] 
    stop_words = stopwords.words('english')
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence


# function for lemmatization 
def lemmatize(x):
    lemmatizer = WordNetLemmatizer()
    return' '.join([lemmatizer.lemmatize(word) for word in x])

# splitting a string, text into a list of tokens
tokenizer = RegexpTokenizer(r'\w+')
def tokenize(x): 
    return tokenizer.tokenize(x)

def sentiment_analysis():
    df = preprocess_data()
    df['combined_text'] = df['combined_text'].map(clean_text)
    df['tokens'] = df['combined_text'].map(tokenize)
    df['tokens'] = df['tokens'].map(remove_stopwords)
    df['lems'] = df['tokens'].map(lemmatize)
    
    # finding the keywords using the rake algorithm from NLTK
    # rake is Rapid Automatic Keyword Extraction algorithm, and is used for domain independent keyword extraction
    df['keywords'] = ""
    for index,row in df.iterrows():
        comb_text = row['combined_text']
        r = Rake()
        r.extract_keywords_from_text(comb_text)
        key_words_dict = r.get_word_degrees()
        row['keywords'] = list(key_words_dict.keys())

    # applying the fucntion to the dataframe
    df['keywords'] = df['keywords'].map(remove_stopwords)
    df['lems'] = df['keywords'].map(lemmatize)
    
    sia = SIA()
    results = []
    for line in df['lems'] :
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)
    
    headlines_polarity = pd.DataFrame.from_records(results)
    # categorize news as positive or negative based on the compound score obtained
    headlines_polarity['label'] = 0
    # I have considered the news as positive if the compound score is greater than 0.2 hence the label 1
    headlines_polarity.loc[headlines_polarity['compound'] > 0.2, 'label'] = 1
    # if the compound score is below 0.2 then it is considered negative 
    headlines_polarity.loc[headlines_polarity['compound'] < -0.2, 'label'] = -1
    # word count of news headlines is calculated
    headlines_polarity['word_count'] = headlines_polarity['headline'].apply(lambda x: len(str(x).split()))
    
    return headlines_polarity

def main():
    query = input("Enter the keyword(s) for news search: ")
    num_results = int(input("Enter the number of results you want: "))
    
    links = get_news_links(query, num_results)
    if not links:
        print("No articles found.")
        return
    
    articles_data = []
    for idx, link in enumerate(links):
        print(f"Fetching article {idx + 1}/{len(links)}: {link}")
        details = get_article_details(link)
        if details:
            articles_data.append(details)
    
    # Create a DataFrame
    df = pd.DataFrame(articles_data)
    print(df)
    
    df = sentiment_analysis()
    print(df.head())

if __name__ == "__main__":
    main()
