import requests
from newspaper import Article
import feedparser
import pandas as pd

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

if __name__ == "__main__":
    main()
