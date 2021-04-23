# abraham

![PyPI](https://img.shields.io/pypi/v/abraham3k)
![PyPI - Downloads](https://img.shields.io/pypi/dm/abraham3k)
![GitHub](https://img.shields.io/github/license/ckinateder/abraham)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/abraham3k)
![GitHub issues](https://img.shields.io/github/issues/ckinateder/abraham)
![GitHub last commit](https://img.shields.io/github/last-commit/ckinateder/abraham)


Algorithmically predict public sentiment on a topic using VADER sentiment analysis.

## Installation

Installation is simple; just install via pip.

```bash
$ pip3 install abraham3k
```

## Basic Usage

You can run one command to execute the main function `news_sentiment_summary`.

```python
from abraham3k.prophets import Isaiah

darthvader = Isaiah(news_source="google") 

scores = darthvader.news_sentiment_summary(["amd", 
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

You can also run a separate function, `summary` to get the raw scores. This will return a nested dictionary with keys for each topic.

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
[87 rows x 6 columns]
      neg    neu    pos  compound                                           sentence              datetime
0   0.200  0.800  0.000   -0.7184  This weekend two men aged 59 69 died Tesla Mod...  2021-04-20T01:12:33Z
1   0.113  0.714  0.172    0.1027  The National Highway Transportation Safety Adm...  2021-04-19T17:11:20Z
2   0.211  0.789  0.000   -0.5859  Tesla working vehicle tailored Chinese consume...  2021-04-20T09:31:36Z
3   0.000  0.702  0.298    0.7003  Amazon told Bloomberg thatit canned Lord Rings...  2021-04-19T11:30:30Z
4   0.128  0.769  0.103   -0.1779  The first fatal incident involve Tesla one dri...  2021-04-19T15:42:47Z
..    ...    ...    ...       ...                                                ...                   ...
76  0.349  0.509  0.142   -0.7717  Two people killed fiery crash Tesla authority ...  2021-04-19T04:02:56Z
77  0.094  0.906  0.000   -0.3818  LiveUpdated April 20, 2021, 11:51 a.m. ET Apri...  2021-04-20T15:36:21Z
78  0.087  0.837  0.076   -0.0754  Though SpaceX‘s new Starlink satellite current...  2021-04-19T08:25:20Z
79  0.275  0.725  0.000   -0.8225  Over weekend, Tesla Model S involved accident ...  2021-04-20T09:50:12Z
80  0.332  0.514  0.154   -0.7096  My heart sink I write AI going wrong. Behind e...  2021-04-20T10:39:02Z
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
