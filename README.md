# abraham

[![PyPI version](https://badge.fury.io/py/abraham3k.svg)](https://badge.fury.io/py/abraham3k)

Algorithmically predict public sentiment on a topic using VADER sentiment analysis.

## Installation

Installation is simple; just install via pip.

```bash
$ pip3 install abraham3k
```

## Basic Usage

You can run one command to do everything

```python
from prophets import Isaiah

darthvader = Isaiah(news_source="google") 

scores = darthvader.sentiment(["amd", 
                               "microsoft", 
                               "tesla", 
                               "theranos"], 
                               window=2)
print(scores)

'''
{'amd': {'avg': 0.32922767,
         'desc_avg': 0.40470959,
         'info': {'news_source': 'google',
                  'splitting': False,
                  'weights': {'desc': 0.1, 'text': 0.8, 'title': 0.1}},
         'nice': 'positive',
         'text_avg': 0.31924348,
         'title_avg': 0.3336193},
 'microsoft': {'avg': 0.22709808,
               'desc_avg': 0.35126282,
               'info': {'news_source': 'google',
                        'splitting': False,
                        'weights': {'desc': 0.1, 'text': 0.8, 'title': 0.1}},
               'nice': 'positive',
               'text_avg': 0.22539444,
               'title_avg': 0.1165625},
 'tesla': {'avg': -0.20538455,
           'desc_avg': -0.22413444,
           'info': {'news_source': 'google',
                    'splitting': False,
                    'weights': {'desc': 0.1, 'text': 0.8, 'title': 0.1}},
           'nice': 'negative',
           'text_avg': -0.19356265,
           'title_avg': -0.28120986},
 'theranos': {'avg': -0.036198,
              'desc_avg': 0.03842,
              'info': {'news_source': 'google',
                       'splitting': False,
                       'weights': {'desc': 0.1, 'text': 0.8, 'title': 0.1}},
              'nice': 'neutral',
              'text_avg': -0.08745,
              'title_avg': 0.2992}}
'''
```

## Changing News Sources

`Isaiah` supports two news sources: [Google News](https://news.google.com/) and [NewsAPI](https://newsapi.org/). Default is [Google News](https://news.google.com/), but you can change it to [NewsAPI](https://newsapi.org/) by passing `Isaiah(news_source='newsapi', api_key='<your api key')` when instantiating. I'd highly recommend using [NewsAPI](https://newsapi.org/). It's much better than the [Google News](https://news.google.com/) API. Setup is really simple, just head to the [register](https://newsapi.org/register) page and sign up to get your API key.

## Detailed Usage

Currently, there are a couple extra options you can use to tweak the output.

When instatiating the class, you can pass up to five optional keyword arguments: `news_source` and `api_key` (as explained above), `splitting`, and `weights`.

* `loud`: `bool` - Whether or not the classifier prints out each individual average or not. Default: `False`.
* `splitting`: `bool` - Recursively splits a large text into sentences and analyzes each sentence individually, rather than examining the article as a block. Default: `False`.
* `weights`: `dict` - This chooses what each individual category (`text`, `title`, `desc`) is weighted as (must add up to 1). Default: `weights={"title": 0.1, "desc": 0.1, "text": 0.8}`.

When running the main function, `sentiment`, there is one requred argument, `topics`, and two optional keyword arguments: `window` and `up_to`.

* `topics`: `list` - The list of the topics (each a `str`) to search for.
* `up_to`: `str` - The latest day to search for, in format `YYYY-MM-DD`. Default: current date.
* `window`: `int` - How many days back from `up_to` to search for. Default `2`.

## Updates

I've made it pretty simple (at least for me) to push updates. Once I'm in the directory, I can run `$ ./build-push 1.2.0 "update install requirements"` where `1.2.0` is the version and `"update install requirements"` is the git commit message. It will update to PyPi and to the github repository.
