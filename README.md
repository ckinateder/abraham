# abraham

![PyPI](https://img.shields.io/pypi/v/abraham3k)
![PyPI - Downloads](https://img.shields.io/pypi/dm/abraham3k)
![GitHub](https://img.shields.io/github/license/ckinateder/abraham)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/abraham3k)
![GitHub issues](https://img.shields.io/github/issues/ckinateder/abraham)
![GitHub last commit](https://img.shields.io/github/last-commit/ckinateder/abraham)


Algorithmically predict public sentiment on a topic using flair sentiment analysis.

## Installation

Installation is simple; just install via pip.

```bash
$ pip3 install abraham3k
```

## Basic Usage

The most simple way of use is to use the `_summary` functions.

```python
from abraham3k.prophets import Isaiah

watched = ["amd", "tesla"]

darthvader = Isaiah(
      news_source="newsapi",
      newsapi_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      bearer_token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      weights={"desc": 0.33, "text": 0.33, "title": 0.34},
)

scores = darthvader.news_summary(
      watched,
      window=2,  # how many days back from up_to to get news from
      up_to="2021-4-22T00:00:00Z",
)
print(scores)

'''
{'amd': (56.2, 43.8), 'tesla': (40.4, 59.6)} # returns a tuple (positive count : negative count)
'''


scores = darthvader.twitter_summary(
      watched,
      start_time="2021-4-20T00:00:00Z" # note the variable name difference from above
      end_time="2021-4-22T00:00:00Z",
)
print(scores)

'''
{'amd': (57, 43), 'tesla': (42, 58)} # returns a tuple (positive count : negative count)
'''
```

You can run the function `news_sentiment` to get the raw scores for the news. This will return a nested dictionary with keys for each topic.

```python
from abraham3k.prophets import Isaiah

darthvader = Isaiah(news_source="google") 

scores = darthvader.news_sentiment(["amd", 
                               "microsoft", 
                               "tesla", 
                               "theranos"], 
                               window=2)
print(scores['tesla']['text'])

'''
                                                 desc              datetime  probability sentiment
0   The latest PassMark ranking show AMD Intel swi...  2021-04-22T18:45:03Z     0.999276  NEGATIVE
1   The X570 chipset AMD offer advanced feature se...  2021-04-22T14:33:07Z     0.999649  POSITIVE
2   Apple released first developer beta macOS 11.4...  2021-04-21T19:10:02Z     0.990774  POSITIVE
3   Prepare terror PC. The release highly anticipa...  2021-04-22T18:00:02Z     0.839055  POSITIVE
4   Stressing ex x86 Canadian AI chip startup Tens...  2021-04-22T13:00:07Z     0.759295  POSITIVE
..                                                ...                   ...          ...       ...
95  Orthopaedic Medical Group Tampa Bay (OMG) exci...  2021-04-21T22:46:00Z     0.979155  POSITIVE
96  OtterBox appointed Leader, proudly 100% Austra...  2021-04-21T23:00:00Z     0.992927  POSITIVE
97  WATG, world's leading global destination hospi...  2021-04-21T22:52:00Z     0.993889  POSITIVE
98  AINQA Health Pte. Ltd. (Headquartered Singapor...  2021-04-22T02:30:00Z     0.641172  POSITIVE
99  Press Release Nokia publish first-quarter repo...  2021-04-22T05:00:00Z     0.894449  NEGATIVE
'''
```

The same way works for the twitter API (see below for integrating twitter usage).

```python
from abraham3k.prophets import Isaiah

darthvader = Isaiah(news_source="google") 

scores = darthvader.twitter_sentiment(["amd", 
                                    "microsoft", 
                                    "tesla", 
                                    "theranos"]
                                    )
```

## Changing News Sources

`Isaiah` supports two news sources: [Google News](https://news.google.com/) and [NewsAPI](https://newsapi.org/). Default is [Google News](https://news.google.com/), but you can change it to [NewsAPI](https://newsapi.org/) by passing `Isaiah(news_source='newsapi', api_key='<your api key')` when instantiating. I'd highly recommend using [NewsAPI](https://newsapi.org/). It's much better than the [Google News](https://news.google.com/) API. Setup is really simple, just head to the [register](https://newsapi.org/register) page and sign up to get your API key.

## Twitter Functionality

I'd highly recommend integrating twitter. It's really simple; just head to [Twitter Developer](https://developer.twitter.com/en) to sign up and get your bearer_token.

## Updates

I've made it pretty simple (at least for me) to push updates. Once I'm in the directory, I can run `$ ./build-push 1.2.0 "update install requirements"` where `1.2.0` is the version and `"update install requirements"` is the git commit message. It will update to PyPi and to the github repository.

## Notes

Currently, there's another algorithm in progress (SALT), including `salt.py` and `salt.ipynb` in the `abraham3k/` directory and the entire `models/` directory. They're not ready for use yet, so don't worry about importing them or anything.

## Contributions

Pull requests welcome!

## Detailed Usage

View the full docstrings here.

```
class Isaiah(builtins.object)
|  Isaiah(news_source='google', newsapi_key=None, bearer_token=None, weights={'title': 0.33, 'desc': 0.33, 'text': 0.34}, loud=False) -> None
|  
|  Performs sentiment analysis on a search term by taking care of gathering
|  all the articles and scoring. Named after the biblical prophet
|  
|  ...
|  
|  Attributes
|  ----------
|  sia : Elijiah
|      Elijiah analyzer
|  news_source : str
|      where to get the news from (google or newsapi)
|  splitting : bool
|      whether or not to recursively analyze each sentence
|  weights : dict
|      how to weight the title, desc, and text attributes
|      ex: {"title": 0.2, "desc": 0.3, "text": 0.5}
|  loud : bool
|      print unnecessary output (for debugging ususally)
|  bearer_token : str
|      bearer token for the twitter api
|  
|  Methods
|  -------
|  get_articles(search_for, up_to=today, window=2)
|      gets articles for a single search term
|  compute_total_avg(results_df, meta)
|      computes avg scores for each row and column of an entire dataframe
|  score_all(topic_results, meta)
|      takes care of scoring the entire dataframe for each topic
|  news_sentiment_summary(topics, window=2, up_to=today)
|      takes a list of topics and computes the avg scores for each
|  news_sentiment(topics, window=2, up_to=today)
|      takes a list of topics and gets the raw scores for each
|      (per topic per text type per row)
|  
|  Methods defined here:
|  
|  __init__(self, news_source='google', newsapi_key=None, bearer_token=None, weights={'title': 0.33, 'desc': 0.33, 'text': 0.34}, loud=False) -> None
|      Parameters
|      ----------
|      news_source : str = "google"
|          where to get the news from
|      newsapi_key : str = None
|          api key to connect to newsapi.org
|      bearer_token : str  = None
|          bearer token for the twitter api
|      spliting : bool = False
|          recursively analyze each sentence or not
|      weights : dict = {"title": 0.33, "desc": 0.33, "text": 0.34}
|          how to weight the title, desc, and text attributes
|      loud : dict = False
|          print unnecessary output (for debugging ususally)
|  
|  get_articles(self, topics: list, window: int = 2, up_to: str = '2021-04-23T21:54:23Z') -> Dict
|      Takes a list of topics and returns a dict of topics : pd.dataframe
|      
|      Parameters
|      ----------
|      topics : list
|          list of terms to search for
|      up_to : str = datetime.now().strftime(TWITTER_TF)
|          latest date to get news for
|      window : int = 2
|          how many days back to search for
|      
|      Returns
|      -------
|      dict
|          in format {topic: <pd.DataFrame>, topic: <pd.DataFrame>, ... } with
|          dataframe being of the results with columns ['title', 'author',
|              'source', 'desc', 'text', 'datetime', 'url', 'urlToImage']
|          ex:
|          {
|              'coinbase': <pd.DataFrame>,
|              'bitcoin': <pd.DataFrame>,
|              ...
|          }
|  
|  news_sentiment(self, topics: list, window: int = 2, up_to: str = '2021-04-23T21:54:23Z')
|      Gets the WHOLE sentiment for each topic. No or minimal averaging occurs.
|      
|      Parameters
|      ----------
|      topics : list
|          list of terms to search for
|      up_to : str = datetime.now().strftime(TWITTER_TF)
|          latest date to get news for
|      window : int = 2
|          how many days back to search for
|      
|      Returns
|      -------
|      scores : dict
|          returns a 2d dict, set up like so:
|          {
|              topic: {"title": titles, "desc": desc, "text": text}
|          }
|          where title, desc, and text are dataframes and each row looks like this:
|          neg    neu    pos  compound                   sentence              datetime
|        0.173  0.827  0.000   -0.5859  Tesla working vehicle ...  2021-04-20T09:31:36Z
|  
|  news_summary(self, topics: list, window: int = 2, up_to: str = '2021-04-23T21:54:23Z')
|      Gets the summary sentiment for each topic
|      
|      Parameters
|      ----------
|      topics : list
|          list of terms to search for
|      up_to : str = datetime.now().strftime(TWITTER_TF)
|          latest date to get news for
|      window : int = 2
|          how many days back to search for
|      
|      Returns
|      -------
|      scores : dict
|          a dict of dicts arranged as {topic: scores},
|          where scores is a tuple (positive count, negative cound)
|  
|  twitter_sentiment(self, topics: list, start_time='2021-04-21T21:54:23Z', end_time='2021-04-23T21:54:23Z')
|      Gets the WHOLE sentiment for each topic. No or minimal averaging occurs.
|      
|      Parameters
|      ----------
|      topics : list
|          list of terms to search for
|      start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
|          how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
|      end_time : str = datetime.now().strftime(TWITTER_TF)
|          how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'
|      
|      Returns
|      -------
|      scores : dict
|          a dict of dataframe of scores for each tweet
|  
|  twitter_summary(self, topics: list, start_time='2021-04-21T21:54:23Z', end_time='2021-04-23T21:54:23Z')
|      Gets the summary sentiment for each topic from twitter
|      
|      Parameters
|      ----------
|      topics : list
|          list of terms to search for
|      start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
|          how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
|      end_time : str = datetime.now().strftime(TWITTER_TF)
|          how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'
|      
|      Returns
|      -------
|      scores : dict
|          a dict of dicts arranged as {topic: scores},
|          where scores is a tuple (positive count, negative cound)

```
