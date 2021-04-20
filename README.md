# abraham

[![PyPI version](https://badge.fury.io/py/abraham3k.svg)](https://badge.fury.io/py/abraham3k)

Algorithmically predict public sentiment on a topic using VADER sentiment analysis.

## Installation

Installation is simple; just install via pip.

```bash
$ pip3 install abraham3k
```

## Sample Output

You can run one command to do everything -

```python
from prophets import Isaiah

# splitting means that it recursively splits a large text into sentences and analyzes each individually
darthvader = Isaiah(news_source="google", splitting=True) 

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

# splitting means that it recursively splits a large text into sentences and analyzes each individually
darthvader = Isaiah(news_source="google", splitting=True)

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

`Isaiah` supports two news sources: [Google News](https://news.google.com/) and [NewsAPI](https://newsapi.org/). Default is [Google News](https://news.google.com/), but you can change it to [NewsAPI](https://newsapi.org/) by passing `Isaiah(news_source='newsapi', api_key='<your api key')` when instantiating.

## NewsAPI Integration

I'd highly recommend using [NewsAPI](https://newsapi.org/). It's much better than the [Google News](https://news.google.com/) API. Setup is really simple, just head to the [register](https://newsapi.org/register) page and sign up. As I explained above, just pass your key to the constructor when instantiating.

## Updates

I've made it pretty simple (at least for me) to push updates. Once I'm in the directory, I can run `$ ./build-push 1.2.0 "update install requirements"` where `1.2.0` is the version and `"update install requirements"` is the git commit message. It will update to PyPi and to the github repository.
