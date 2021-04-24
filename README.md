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

## Detailed Usage

Currently, there are a couple extra options you can use to tweak the output.

When instatiating the class, you can pass up to five optional keyword arguments: `news_source` and `api_key` (as explained above), `splitting`, and `weights`.

* `loud`: `bool` - Whether or not the classifier prints out each individual average or not. Default: `False`.
* `splitting`: `bool` - Recursively splits a large text into sentences and analyzes each sentence individually, rather than examining the article as a block. Default: `False`.
* `weights`: `dict` - This chooses what each individual category (`text`, `title`, `desc`) is weighted as (must add up to 1). Default: `weights={"title": 0.1, "desc": 0.1, "text": 0.8}`.

When running the main functions, `news_sentiment` and `news_sentiment_summary`, there is one requred argument, `topics`, and two optional keyword arguments: `window` and `up_to`.

* `topics`: `list` - The list of the topics (each a `str`) to search for.
* `up_to`: `str` - The latest day to search for, in ISO format (`%Y-%m-%dT%H:%M:%SZ`). Default: current date.
* `window`: `int` - How many days back from `up_to` to search for. Default `2`.

## Updates

I've made it pretty simple (at least for me) to push updates. Once I'm in the directory, I can run `$ ./build-push 1.2.0 "update install requirements"` where `1.2.0` is the version and `"update install requirements"` is the git commit message. It will update to PyPi and to the github repository.

## Notes

Currently, there's another algorithm in progress (SALT), including `salt.py` and `salt.ipynb` in the `abraham3k/` directory and the entire `models/` directory. They're not ready for use yet, so don't worry about importing them or anything.

## Contributions

Pull requests welcome!
