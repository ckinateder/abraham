# abraham

Algorithmically predict public sentiment on a topic using VADER sentiment analysis.

## Sample Output

You can run one command to do everything -

```python
from prophets import Isaiah
darthvader = Isaiah(news_source="google", splitting=True) # splitting means that it recursively splits a large text into sentences and analyzes each individually

# this command takes a bit of time to run because it has to download lots of articles
scores = darthvader.sentiment(["robinhood", 
                      "johnson and johnson", 
                      "bitcoin", 
                      "dogecoin", 
                      "biden",  
                      "amazon"], 
                      window=2, # how many days back from up_to to get news from
                      up_to="04/18/2021") # latest date to get news from

print(scores)

'''
{'robinhood': 
    {
        'avg': 0.3798676562301132, 
        'nice': 'positive :)'
     },
 'johnson and johnson': 
    {
        'avg': 0.27466788299009787, 
        'nice': 'positive :)'
    },
 'bitcoin': 
    {
        'avg': 0.28669931035859125, 
        'nice': 'positive :)'
    },
 'dogecoin': 
    {
        'avg': 0.2837840361036227, 
        'nice': 'positive :)'
    },
 'biden': 
    {
        'avg': 0.2404157345348728, 
        'nice': 'positive :)'
    },
 'amazon': 
    {
        'avg': 0.2894022880254384, 
        'nice': 'positive :)'
    }
}
'''
```

Or, you can run it step by step, as well.

```python
from prophets import Isaiah
darthvader = Isaiah(news_source="google", splitting=True) # splitting means that it recursively splits a large text into sentences and analyzes each individually

# this command takes a bit of time to run because it has to download lots of articles
articles = darthvader.get_articles(["robinhood", 
                      "johnson and johnson", 
                      "bitcoin", 
                      "dogecoin", 
                      "biden",  
                      "amazon"]
                      window=2, # how many days back from up_to to get news from
                      up_to="04/18/2021") # latest date to get news from

scores = darthvader.score_all(articles)

print(scores)

'''
{'robinhood': 
    {
        'avg': 0.3798676562301132, 
        'nice': 'positive :)'
     },
 'johnson and johnson': 
    {
        'avg': 0.27466788299009787, 
        'nice': 'positive :)'
    },
 'bitcoin': 
    {
        'avg': 0.28669931035859125, 
        'nice': 'positive :)'
    },
 'dogecoin': 
    {
        'avg': 0.2837840361036227, 
        'nice': 'positive :)'
    },
 'biden': 
    {
        'avg': 0.2404157345348728, 
        'nice': 'positive :)'
    },
 'amazon': 
    {
        'avg': 0.2894022880254384, 
        'nice': 'positive :)'
    }
}
'''
```

`Isaiah` supports two news sources: [Google News]([google news](https://news.google.com/)) and [NewsAPI](https://newsapi.org/). Default is [Google News]([google news](https://news.google.com/)), but you can change it to [NewsAPI](https://newsapi.org/) by passing `Isaiah(news_source='newsapi')` when instantiating. In order to use NewsAPI, you have to put your api key in `keys/newsapi_org`.
