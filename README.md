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
      start_time="2021-4-20T00:00:00Z" 
      end_time="2021-4-22T00:00:00Z",
)
print(scores)

'''
{'amd': (56.2, 43.8), 'tesla': (40.4, 59.6)} # returns a tuple (positive count : negative count)
'''


scores = darthvader.twitter_summary(
      watched,
      start_time="2021-4-20T00:00:00Z" 
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
from abraham3k.prophets import Isaiah

darthvader = Isaiah(news_source="google") 

scores = darthvader.twitter_sentiment(["amd", 
                                    "microsoft", 
                                    "tesla", 
                                    "theranos"]
                                    )
```

You can also just use a one-off function to get the sentiment from both the news and twitter combined.

```python
from abraham3k.prophets import Isaiah

darthvader = Isaiah(news_source="google") 

scores = darthvader.summary(["tesla", "amd"], weights={"news": 0.5, "twitter": 0.5})

print(scores)
'''
{'amd': (59.0, 41.0), 'tesla': (46.1, 53.9)}
'''
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

Coming soon. However, there is heavy documentation in the actual code.
