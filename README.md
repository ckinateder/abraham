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
from abraham3k.prophets import Abraham
from datetime import datetime, timedelta

watched = ["amd", "tesla"]

darthvader = Abraham(
      news_source="newsapi",
      newsapi_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      bearer_token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      weights={"desc": 0.33, "text": 0.33, "title": 0.34},
)

scores = darthvader.news_summary(
      watched,
      start_time=datetime.now() - timedelta(days=1) 
      end_time=datetime.now(),
)
print(scores)

'''
{'amd': (56.2, 43.8), 'tesla': (40.4, 59.6)} # returns a tuple (positive count : negative count)
'''

scores = darthvader.twitter_summary(
      watched,
      start_time=datetime.now() - timedelta(days=1) 
      end_time=datetime.now(),
)
print(scores)

'''
{'amd': (57, 43), 'tesla': (42, 58)} # returns a tuple (positive count : negative count)
'''
```

You can run the function `news_sentiment` to get the raw scores for the news. This will return a nested dictionary with keys for each topic.

```python
from abraham3k.prophets import Abraham
from datetime import datetime, timedelta

darthvader = Abraham(news_source="google") 

scores = darthvader.news_sentiment(["amd", 
                               "microsoft", 
                               "tesla", 
                               "theranos"],
                               )
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
from abraham3k.prophets import Abraham
from datetime import datetime, timedelta

darthvader = Abraham(news_source="google") 

scores = darthvader.twitter_sentiment(["amd", 
                                    "microsoft", 
                                    "tesla", 
                                    "theranos"]
                                    )
```

You can also just use a one-off function to get the sentiment from both the news and twitter combined.

```python
from abraham3k.prophets import Abraham
from datetime import datetime, timedelta

darthvader = Abraham(news_source="google") 

scores = darthvader.summary(["tesla", "amd"], weights={"news": 0.5, "twitter": 0.5})

print(scores)

'''
{'amd': (59.0, 41.0), 'tesla': (46.1, 53.9)}
'''
```

There's also a built-in function for building a dataset of past sentiments. This follows the same format as the non-interval functions (`twitter_summary_interval`, `news_summary_interval`, `summary_interval`).

```python
from abraham3k.prophets import Abraham
from datetime import datetime, timedelta

# this works best using the offical twitter api rather than twint
darthvader = Abraham(bearer_token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx") 

scores = twitter_summary_interval(
        self,
        ["tesla", "amd"],
        oldest=datetime.now() - timedelta(days=1),
        newest=datetime.now(),
        interval=timedelta(hours=12),
        offset=timedelta(hours=1),
        size=100,
    )

print(scores)

'''
                    timestamp  positive  negative             lag
0  2021-05-08 11:46:57.033549      61.0      39.0 0 days 12:00:00
1  2021-05-08 10:46:57.033549      54.0      46.0 0 days 12:00:00
2  2021-05-08 09:46:57.033549      68.0      32.0 0 days 12:00:00
3  2021-05-08 08:46:57.033549      78.0      22.0 0 days 12:00:00
4  2021-05-08 07:46:57.033549      71.0      29.0 0 days 12:00:00
5  2021-05-08 06:46:57.033549      74.0      26.0 0 days 12:00:00
6  2021-05-08 05:46:57.033549      63.0      37.0 0 days 12:00:00
7  2021-05-08 04:46:57.033549      74.0      26.0 0 days 12:00:00
8  2021-05-08 03:46:57.033549      53.5      46.5 0 days 12:00:00
9  2021-05-08 02:46:57.033549      51.0      49.0 0 days 12:00:00
10 2021-05-08 01:46:57.033549      61.0      39.0 0 days 12:00:00
11 2021-05-08 00:46:57.033549      46.9      53.1 0 days 12:00:00
12 2021-05-07 23:46:57.033549      54.0      46.0 0 days 12:00:00
13 2021-05-07 22:46:57.033549      52.0      48.0 0 days 12:00:00
14 2021-05-07 21:46:57.033549      58.0      42.0 0 days 12:00:00
15 2021-05-07 20:46:57.033549      46.0      54.0 0 days 12:00:00
16 2021-05-07 19:46:57.033549      40.0      60.0 0 days 12:00:00
17 2021-05-07 18:46:57.033549      40.0      60.0 0 days 12:00:00
18 2021-05-07 17:46:57.033549      51.0      49.0 0 days 12:00:00
19 2021-05-07 16:46:57.033549      21.0      79.0 0 days 12:00:00
20 2021-05-07 15:46:57.033549      52.5      47.5 0 days 12:00:00
21 2021-05-07 14:46:57.033549      36.0      64.0 0 days 12:00:00
22 2021-05-07 13:46:57.033549      42.0      58.0 0 days 12:00:00
23 2021-05-07 12:46:57.033549      40.0      60.0 0 days 12:00:00
24 2021-05-07 11:46:57.033549      32.0      68.0 0 days 12:00:00
'''
```

Google trends is also in the process of being added. Currently, there's support for interest over time. You can access it like this.

```python
from abraham3k.prophets import Abraham
from datetime import datetime, timedelta

darthvader = Abraham()

results = darthvader.interest_interval(
        ["BTC USD", "buy bitcoin"],
        start_time=(datetime.now() - timedelta(days=52)),
        end_time=datetime.now())

print(results)

'''
            BTC USD  buy bitcoin
date                            
2021-03-24       62           18
2021-03-25       68           16
2021-03-26       58           12
2021-03-27       47           15
2021-03-28       48           15
...
2021-05-08       48           27
2021-05-09       38           25
2021-05-10       43           20
2021-05-11       44           24
2021-05-12       38           20
'''
```

Numbers represent search interest relative to the highest point on the chart for the given region and time. A value of 100 is the peak popularity for the term. A value of 50 means that the term is half as popular. A score of 0 means there was not enough data for this term.

## Changing News Sources

`Abraham` supports two news sources: [Google News](https://news.google.com/) and [NewsAPI](https://newsapi.org/). Default is [Google News](https://news.google.com/), but you can change it to [NewsAPI](https://newsapi.org/) by passing `Abraham(news_source='newsapi', api_key='<your api key')` when instantiating. I'd highly recommend using [NewsAPI](https://newsapi.org/). It's much better than the [Google News](https://news.google.com/) API. Setup is really simple, just head to the [register](https://newsapi.org/register) page and sign up to get your API key.

## Twitter Functionality

I'd highly recommend integrating twitter. It's really simple; just head to [Twitter Developer](https://developer.twitter.com/en) to sign up and get your bearer token. If you don't want to sign up, you can actually use it free with the twint API (no keys needed). This is the default.

## Updates

I've made it pretty simple (at least for me) to push updates. Once I'm in the directory, I can run `$ ./build-push 1.2.0 "update install requirements"` where `1.2.0` is the version and `"update install requirements"` is the git commit message. It will update to PyPi and to the github repository.

## Notes

Currently, there's another algorithm in progress (SALT), including `salt.py` and `salt.ipynb` in the `abraham3k/` directory and the entire `models/` directory. They're not ready for use yet, so don't worry about importing them or anything.

## Contributions

Pull requests welcome!

## Detailed Usage

Coming soon. However, there is heavy documentation in the actual code.
