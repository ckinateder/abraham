from typing import Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from GoogleNews import GoogleNews
from newspaper import Article, ArticleException
from tqdm import tqdm
from threading import Thread
from datetime import datetime, timedelta
import re
import pandas as pd
import time, requests
import warnings
import sys
from dateutil.parser._parser import UnknownTimezoneWarning
import flair

warnings.simplefilter("ignore", UnknownTimezoneWarning)

# define time formats for each
GOOGLENEWS_TF = "%m/%d/%Y"
# TWITTER_TF = "%Y-%m-%d"
TWITTER_TF = "%Y-%m-%dT%H:%M:%SZ"

NEWSAPI_URL = "https://newsapi.org/v2/everything?"
TWITTER_URL = "https://api.twitter.com/2/tweets/search/recent"


class NewsAPI:
    """
    A class used to fetch news from newsapi.org

    ...

    Attributes
    ----------
    newsapi_key : str
        api key to connect to newsapi.org

    Methods
    -------
    fetch_json(searchfor, url=NEWSAPI_URL, pagesize=100, page=1,
               language="en", from_date=7 days ago, to_date=today)
        fetches the articles and returns a json
    clean_response(jsonfile)
        cleans the fetched jsonfile and renames columns
    cleaned_to_df(cleaned_dict)
        parses the cleaned json dictionary to a pandas DataFrame
    get_articles(search_for, up_to=today, window=1)
        wraps the 3 previous functions all into one user friendly one
    """

    def __init__(
        self,
        newsapi_key,
    ) -> None:
        """
        Parameters
        ----------
        newsapi_key : str
            api key to connect to newsapi.org
        """
        self.newsapi_key = newsapi_key

    def fetch_json(
        self,
        searchfor: str,
        url: str = NEWSAPI_URL,
        pagesize: int = 100,
        page: int = 1,
        language: str = "en",
        from_date: str = (datetime.now() - timedelta(7)).strftime(TWITTER_TF),
        to_date: str = datetime.now().strftime(TWITTER_TF),
    ):
        """Search the news for a search term.

        Parameters
        ----------
        searchfor: str
            term to search for
        url: str = NEWSAPI_URL, optional
            url to pass calls to
        pagesize: int = 100, optional
            pagesize to return the object
        page: int = 1, optional
            page to read from
        language: str = "en", optional
            language to search in
        from_date: str = (datetime.now() - timedelta(7)).strftime(TWITTER_TF), optional
            farthest day back to go when searching, in "%Y-%m-%d" format
        to_date: str = datetime.now().strftime(TWITTER_TF), optional
            latest date to go up to when searching, in "%Y-%m-%d" format

        Returns
        -------
        json
            a raw json response containing just the articles and their data
        """
        params = {
            "q": searchfor,
            "pageSize": pagesize,
            "apiKey": self.newsapi_key,
            "language": language,
            "page": page,
            "from": from_date,
            "to": to_date,
        }
        response = requests.get(url, params=params)
        json_response = response.json()["articles"]
        return json_response

    def clean_response(self, jsonfile):
        """Cleans up a json response and gives a dictionary

        Parameters
        ----------
        searchfor: json
            raw json to be cleaned

        Returns
        -------
        list
            a list of every single cleaned dictionary
        """
        results = []
        for i in range(len(jsonfile)):
            cleaned_item = {}
            cleaned_item["title"] = jsonfile[i]["title"]
            cleaned_item["author"] = jsonfile[i]["author"]
            cleaned_item["source"] = jsonfile[i]["source"]
            cleaned_item["desc"] = jsonfile[i]["description"]
            cleaned_item["text"] = jsonfile[i]["content"]
            cleaned_item["datetime"] = jsonfile[i]["publishedAt"]
            cleaned_item["url"] = jsonfile[i]["url"]
            cleaned_item["urlToImage"] = jsonfile[i]["urlToImage"]
            results.append(cleaned_item)

        return results

    def cleaned_to_df(self, cleaned_dict):
        """Creates a pandas DataFrame from a cleaned dictionary

        Parameters
        ----------
        cleaned_dict: list
            list of dicts to be converted

        Returns
        -------
        pandas.DataFrame
            a dataframe of the results with columns ['title', 'author', 'source', 'desc',
                                                    'text', 'datetime', 'url', 'urlToImage']
        """
        return pd.DataFrame(cleaned_dict)

    def get_articles(
        self, searchfor, up_to=datetime.now().strftime(TWITTER_TF), window=1
    ):
        """Gets articles for a single search term

        Parameters
        ----------
        searchfor: str
            term to search for
        up_to : str = datetime.now().strftime(TWITTER_TF)
            latest date to get news for, in "%Y-%m-%d" format
        window : int = 1
            how many days back to search for

        Returns
        -------
        pandas.DataFrame
            a dataframe of the results with columns ['title', 'author', 'source', 'desc',
                                                    'text', 'datetime', 'url', 'urlToImage']
        """
        period = (datetime.now() - timedelta(window)).strftime(TWITTER_TF)
        jresponse = self.fetch_json(searchfor, from_date=period, to_date=up_to)
        cleaned = self.clean_response(jresponse)
        cleaned_df = self.cleaned_to_df(cleaned)
        return cleaned_df


class GoogleNewsParser:
    """
    A class used to fetch news from googlenews

    ...

    Attributes
    ----------
    None

    Methods
    -------
    _get_text(inst)
        gets the text for each article it recieves
    get_articles(search_for, up_to=today, window=1)
        Gets articles for a single search term
    """

    def __init__(self) -> None:
        self.googlenews = GoogleNews()  # create news object

    def _get_text(self, inst):
        """Gets text from an article url (inst["link"])

        Parameters
        ----------
        inst : dict
            dictionary containing the link

        Returns
        -------
        str
            the article
        """
        try:
            article = Article(
                "http://"
                + inst["link"]
                .replace("http://", "")
                .replace("https://", "")  # remove https that already exists
            )
            article.download()
            article.parse()
            text = article.text.strip().replace("\n", " ")
            inst["text"] = text

        except ArticleException:
            inst["text"] = ""
        self.pbar.update(1)
        return inst["text"]

    def get_articles(
        self,
        search_term,
        up_to=datetime.now().strftime(TWITTER_TF),
        window=2,  # how many days back to go
    ):
        """Gets articles for a single search term

        Parameters
        ----------
        searchfor: str
            term to search for
        up_to : str = datetime.now().strftime(TWITTER_TF)
            latest date to get news for
        window : int = 1
            how many days back to search for

        Returns
        -------
        pandas.DataFrame
            a dataframe of the results with columns ['title', 'author', 'source', 'desc',
                                                    'text', 'datetime', 'url', 'urlToImage']
        """
        start = time.time()
        # use settimerange instead
        end_date = datetime.strptime(up_to, TWITTER_TF).strftime(GOOGLENEWS_TF)
        start_date = (datetime.now() - timedelta(window)).strftime(GOOGLENEWS_TF)
        self.googlenews.set_time_range(start_date, end_date)  # set the range
        self.googlenews.get_news(search_term)  # get the news
        results = self.googlenews.results()  # get the results

        self.pbar = tqdm(
            total=len(results), unit="article", desc=search_term, leave=False
        )

        processes = []  # multi thread the execution
        for i in results:
            processes.append(
                Thread(
                    target=self._get_text,
                    args=(i,),
                )
            )

        # start
        for proc in processes:
            proc.start()
        # join
        for proc in processes:
            proc.join()
        self.pbar.close()

        # print(f"Got {len(results)} articles in {time.time()-start:.2f}s")
        return pd.DataFrame(results)


class TwitterParser:
    """
    Gets tweets for analyzing

    ...

    Attributes
    ----------
    bearer_token : str
        twitter api bearer token

    Methods
    .......
    get_tweets(topic)
        takes a topic and gets the tweets for it
    """

    def __init__(self, bearer_token) -> None:
        """
        Parameters
        ----------
        bearer_token : str
            twitter api bearer token
        """
        self.bearer_token = bearer_token

    def parse_tweet(self, tweet):
        """Parse a tweet and return just what we need

        Parameters
        ----------
        tweet : json
            the raw tweet object

        Returns
        -------
        data : dict
            a dict of just the important parts (id, text, created_at)
        """
        data = {
            "id": tweet["id"],
            "created_at": tweet["created_at"],
            "text": tweet["text"],
        }
        return data

    def get_tweets(
        self,
        topic,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
    ):  # how many days back to go
        """Get the tweets for a given topic

        Parameters
        ----------
        topic : str
            topic to search for
        start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
            how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
        end_time : str = datetime.now().strftime(TWITTER_TF)
            how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'

        Returns
        -------
        tweets : pd.DataFrame
            a dataframe of the tweets
            Sample row:
                             created_at        id                        text
        Fri Apr 23 17:44:44 +0000 2021  138565089  RT @jenine1207: @siiyuun...
        """
        # define params
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "query": f"({topic}) (lang:en)",
            "max_results": "100",
            "tweet.fields": "created_at,lang",
        }
        response = requests.get(
            TWITTER_URL,
            params=params,
            headers={"authorization": "Bearer " + self.bearer_token},
        )
        tweets = pd.DataFrame()
        if response.status_code == 200:
            for tweet in tqdm(
                response.json()["data"],
                leave=False,
                desc=f"{topic} tweets",
                dynamic_ncols=True,
            ):
                try:
                    row = self.parse_tweet(tweet)
                    tweets = tweets.append(row, ignore_index=True)
                except Exception as e:
                    warnings.warn(f"Error while parsing tweet ... skipping ({e})")
        else:
            warnings.warn(f"Response code {response.status_code} recieved")

        return tweets


class Elijiah:
    """
    Performs sentiment analysis on sentences. Named after the biblical prophet

    ...

    Attributes
    ----------
    vader : SentimentIntensityAnalyzer
        VADER analyzer from nltk for the news
    sunflair : flair.models.TextClassifier
        flair model for the tweets
    lemmatizer : WordNetLemmatizer
        lemmatizer from nltk

    Methods
    -------
    sentiment_analyzer_sent(sentence)
        analyzes a single sentence and returns the score
    normalize_text(sentence)
        normalizes a sentence (removes tags, lemmatizes, etc...)
    analyze_news_text(frame, section, recursive)
        takes a dataframe and a section and scores each row. if recursive=True, it will only
        analyze one sentence at time
    """

    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        print("Importing Flair")
        self.sunflair = flair.models.TextClassifier.load("en-sentiment")

    def sentiment_analyzer_sent(self, sentence: str):
        """Analyzes a single sentence and returns the score

        Parameters
        ----------
        sentence : str
            sentence to analyze

        Returns
        -------
        dict
            the scores
        """
        score = self.vader.polarity_scores(sentence)
        print("{:-<40} {}".format(sentence, str(score)))
        return score

    def normalize_text(self, sentence: str):
        """Normalizes a sentence

        Parameters
        ----------
        sentence : str
            sentence to normalize

        Returns
        -------
        str
            the normalized sentetnce
        """
        subbed = sentence
        clean = re.compile("<.*?>")

        web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
        user = re.compile(r"(?i)@[a-z0-9_]+")
        whitespace = re.compile(r"\s+")

        subbed = web_address.sub("", subbed)
        # subbed = whitespace.sub("", subbed)
        subbed = user.sub("", subbed)
        subbed = re.sub(clean, "", subbed)
        subbed = re.sub("\[+(.*?)chars\]", "", subbed)
        word_list = subbed.split()
        filtered_words = [
            word for word in word_list if word not in stopwords.words("english")
        ]
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
        new_sentence = " ".join(lemmatized_words)
        return new_sentence

    def analyze_news_text(
        self, frame: pd.DataFrame, section: str, recursive: bool = False
    ):
        """Takes a dataframe and a section and scores each row of frame[section]. If recursive=True, it will only
        analyze one sentence at time.
        Example -
            neg    neu    pos  compound                   sentence
        0  0.123  0.877  0.000   -0.1027   Cramer call Coinbase 'real deal,' warns invest
        ...

        Parameters
        ----------
        frame: pd.DataFrame
            dataframe containg section
        section: str
            section of dataframe to analyze
        recursive: bool = False

        Returns
        -------
        pandas.DataFrame
            new dataframe of scores
        """
        scores = []
        for ind in tqdm(
            frame.index, leave=False, unit="text", desc=f"analyze {section}"
        ):
            # iterate through list of sent tokenize
            item = frame[section][ind]
            date = frame["datetime"][ind]

            if recursive:
                for sentence in sent_tokenize(item):
                    sentence = self.normalize_text(sentence)
                    score = self.vader.polarity_scores(sentence)
                    if score["compound"] != 0:  # remove all zeroes
                        score["sentence"] = sentence
                        score["datetime"] = date
                        scores.append(score)
            else:
                sentence = self.normalize_text(item)
                score = self.vader.polarity_scores(sentence)
                if score["compound"] != 0:  # remove all zeroes
                    score["sentence"] = sentence
                    score["datetime"] = date
                    scores.append(score)
        scores = pd.DataFrame(scores)
        return scores

    def analyze_tweet_text(self, tweets: pd.DataFrame):
        """Takes a dataframe of tweets and analyzes and saves the score for each row
        ...

        Parameters
        ----------
        tweet_frame: pd.DataFrame
            dataframe containg section

        Returns
        -------
        pandas.DataFrame
            new dataframe of scores
        """
        # we will append probability and sentiment preds later
        probs = []
        sentiments = []

        tweets["text"] = tweets["text"].apply(self.normalize_text)

        for tweet in tweets["text"].to_list():
            # make prediction
            sentence = flair.data.Sentence(tweet)
            self.sunflair.predict(sentence)
            # extract sentiment prediction
            probs.append(sentence.labels[0].score)  # numerical score 0-1
            sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'

        # add probability and sentiment predictions to tweets dataframe
        tweets["probability"] = probs
        tweets["sentiment"] = sentiments
        return tweets


class Isaiah:
    """
    Performs sentiment analysis on a search term by taking care of gathering
    all the articles and scoring. Named after the biblical prophet

    ...

    Attributes
    ----------
    sia : Elijiah
        Elijiah analyzer
    news_source : str
        where to get the news from (google or newsapi)
    splitting : bool
        whether or not to recursively analyze each sentence
    weights : dict
        how to weight the title, desc, and text attributes
        ex: {"title": 0.2, "desc": 0.3, "text": 0.5}
    loud : bool
        print unnecessary output (for debugging ususally)
    bearer_token : str
        bearer token for the twitter api

    Methods
    -------
    get_articles(search_for, up_to=today, window=2)
        gets articles for a single search term
    compute_total_avg(results_df, meta)
        computes avg scores for each row and column of an entire dataframe
    score_all(topic_results, meta)
        takes care of scoring the entire dataframe for each topic
    news_sentiment_summary(topics, window=2, up_to=today)
        takes a list of topics and computes the avg scores for each
    news_sentiment(topics, window=2, up_to=today)
        takes a list of topics and gets the raw scores for each
        (per topic per text type per row)
    """

    def __init__(
        self,
        news_source="google",
        newsapi_key=None,
        bearer_token=None,
        splitting=False,
        weights={"title": 0.2, "desc": 0.3, "text": 0.5},
        loud=False,
    ) -> None:
        """
        Parameters
        ----------
        news_source : str = "google"
            where to get the news from
        newsapi_key : str = None
            api key to connect to newsapi.org
        bearer_token : str  = None
            bearer token for the twitter api
        spliting : bool = False
            recursively analyze each sentence or not
        weights : dict = {"title": 0.2, "desc": 0.3, "text": 0.5}
            how to weight the title, desc, and text attributes
        loud : dict = False
            print unnecessary output (for debugging ususally)
        """
        self.sia = Elijiah()
        if news_source == "newsapi":
            if newsapi_key:
                self.news_source = "newsapi"
                self.newsparser = NewsAPI(newsapi_key=newsapi_key)
            else:
                print(
                    "You requested newsapi but no key was provided. Defaulting to googlenews."
                )
                self.news_source = "google"
                self.newsparser = GoogleNewsParser()
        else:
            self.news_source = "google"
            self.newsparser = GoogleNewsParser()
        self.splitting = splitting  # does sia.analyze use recursion to split?
        self.weights = weights
        self.loud = loud
        self.bearer_token = bearer_token

    def get_articles(
        self,
        topics: list,
        window: int = 2,
        up_to: str = datetime.now().strftime(TWITTER_TF),
    ) -> Dict:
        """Takes a list of topics and returns a dict of topics : pd.dataframe

        Parameters
        ----------
        topics : list
            list of terms to search for
        up_to : str = datetime.now().strftime(TWITTER_TF)
            latest date to get news for
        window : int = 2
            how many days back to search for

        Returns
        -------
        dict
            in format {topic: <pd.DataFrame>, topic: <pd.DataFrame>, ... } with
            dataframe being of the results with columns ['title', 'author',
                'source', 'desc', 'text', 'datetime', 'url', 'urlToImage']
            ex:
            {
                'coinbase': <pd.DataFrame>,
                'bitcoin': <pd.DataFrame>,
                ...
            }
        """
        topic_results = {}
        for topic in topics:
            topic_results[topic] = self.newsparser.get_articles(
                topic, window=window, up_to=up_to
            )
        return topic_results

    def compute_total_avg(self, results_df: pd.DataFrame, meta: dict):
        """Computes avg scores for each row and then column of an entire dataframe

        Parameters
        ----------
        results_df: pd.DataFrame
            dataframe of results with columns ['title', 'author', 'source',
                            'desc', 'text', 'datetime', 'url', 'urlToImage']
        meta: dict
            any additional information to return with the scores.
            usually {"window": window, "up_to": up_to}

        Returns
        -------
        returned_dict : dict
            a dict of the avg scores and meta information
            ex:
            {
                'avg': -0.06751676,
                'desc_avg': -0.07466768,
                'info': {
                    'news_source': 'newsapi',
                    'splitting': True,
                    'up_to': '2021-4-20',
                    'weights': {'desc': 0.3, 'text': 0.5, 'title': 0.2},
                    'window': 2
                        },
                'nice': 'negative',
                'text_avg': -0.04153505,
                'title_avg': -0.12174464
            }

        """
        title_avg = round(
            self.sia.analyze_news_text(
                results_df, "title", recursive=self.splitting
            ).compound.mean(),
            8,
        )
        desc_avg = round(
            self.sia.analyze_news_text(
                results_df, "desc", recursive=self.splitting
            ).compound.mean(),
            8,
        )
        text_avg = round(
            self.sia.analyze_news_text(
                results_df, "text", recursive=self.splitting
            ).compound.mean(),
            8,
        )

        if self.weights["title"] + self.weights["text"] + self.weights["desc"] != 1:
            wstr = f"WARNING: Sum of custom weights != 1 ({self.weights})"
            warnings.warn(wstr)

        total_avg = round(
            title_avg * self.weights["title"]
            + desc_avg * self.weights["desc"]
            + text_avg * self.weights["text"],
            8,
        )
        # classify
        if total_avg >= 0.05:
            classified = "positive"
        elif total_avg <= -0.05:
            classified = "negative"
        else:
            classified = "neutral"

        if self.loud:
            print(f"Title avg: {round(title_avg*100,2)}% (compound={title_avg:.8f})")
            print(f"Desc avg: {round(desc_avg*100,2)}% (compound={desc_avg:.8f})")
            print(f"Text avg: {round(text_avg*100,2)}% (compound={text_avg:.8f})")
            print(
                f"[Total avg: {round(total_avg*100,2)}% (compound={total_avg:.8f}) <{classified}>]"
            )

        meta["weights"] = self.weights
        meta["news_source"] = self.news_source
        meta["splitting"] = self.splitting

        returned_dict = {
            "avg": total_avg,
            "title_avg": title_avg,
            "desc_avg": desc_avg,
            "text_avg": text_avg,
            "info": meta,
            "nice": classified,
        }
        return returned_dict

    def score_all(self, topic_results: dict, meta: dict):
        """Takes care of scoring the entire dataframe for each topic

        Paramaters
        ----------
        topic_results: dict
            in format {topic: <pd.DataFrame>, topic: <pd.DataFrame>, ... } with
            dataframe being of the results with columns ['title', 'author',
                'source', 'desc', 'text', 'datetime', 'url', 'urlToImage']
            ex:
            {
                'coinbase': <pd.DataFrame>,
                'bitcoin': <pd.DataFrame>,
                ...
            }
        meta: dict
            any additional information to return with the scores.
            usually {"window": window, "up_to": up_to}

        Returns
        -------
        scores : dict
            a dict of dicts arranged as {topic: scores}
            ex:
            {
            'amd': {'avg': 0.2880456,
                    'desc_avg': 0.31842738,
                    'info': {
                        'news_source': 'newsapi',
                        'splitting': True,
                        'up_to': '2021-4-20',
                        'weights': {'desc': 0.3, 'text': 0.5, 'title': 0.2},
                        'window': 2
                            },
                    'nice': 'positive',
                    'text_avg': 0.2613019,
                    'title_avg': 0.3093322},
            'tesla': {'avg': -0.06751676,
                    'desc_avg': -0.07466768,
                    'info': {
                        'news_source': 'newsapi',
                        'splitting': True,
                        'up_to': '2021-4-20',
                        'weights': {'desc': 0.3, 'text': 0.5, 'title': 0.2},
                        'window': 2
                            },
                    'nice': 'negative',
                    'text_avg': -0.04153505,
                    'title_avg': -0.12174464},
            }

        """
        scores = {}
        for topic in topic_results:
            classification = self.compute_total_avg(topic_results[topic], meta=meta)
            scores[topic] = classification
        return scores

    def news_sentiment_summary(
        self,
        topics: list,
        window: int = 2,
        up_to: str = datetime.now().strftime(TWITTER_TF),
    ):
        """Gets the summary sentiment for each topic

        Parameters
        ----------
        topics : list
            list of terms to search for
        up_to : str = datetime.now().strftime(TWITTER_TF)
            latest date to get news for
        window : int = 2
            how many days back to search for

        Returns
        -------
        scores : dict
            a dict of dicts arranged as {topic: scores} (see score_all for a sample return)
        """
        articles = self.get_articles(topics, window=window, up_to=up_to)
        scores = self.score_all(articles, meta={"window": window, "up_to": up_to})
        return scores

    def news_sentiment(
        self,
        topics: list,
        window: int = 2,
        up_to: str = datetime.now().strftime(TWITTER_TF),
    ):
        """Gets the WHOLE sentiment for each topic. No or minimal averaging occurs.

        Parameters
        ----------
        topics : list
            list of terms to search for
        up_to : str = datetime.now().strftime(TWITTER_TF)
            latest date to get news for
        window : int = 2
            how many days back to search for

        Returns
        -------
        scores : dict
            returns a 2d dict, set up like so:
            {
                topic: {"title": titles, "desc": desc, "text": text}
            }
            where title, desc, and text are dataframes and each row looks like this:
            neg    neu    pos  compound                   sentence              datetime
          0.173  0.827  0.000   -0.5859  Tesla working vehicle ...  2021-04-20T09:31:36Z
        """

        articles = self.get_articles(topics, window=window, up_to=up_to)

        scores = {}
        for topic in articles:
            titles = self.sia.analyze_news_text(
                articles[topic], "title", recursive=self.splitting
            )
            desc = self.sia.analyze_news_text(
                articles[topic], "desc", recursive=self.splitting
            )
            text = self.sia.analyze_news_text(
                articles[topic], "text", recursive=self.splitting
            )
            scores[topic] = {"title": titles, "desc": desc, "text": text}
        return scores

    def twitter_sentiment(
        self,
        topics: list,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
    ):
        """Gets the WHOLE sentiment for each topic. No or minimal averaging occurs.

        Parameters
        ----------
        topics : list
            list of terms to search for

        Returns
        -------
        scores : dict
            a dict of dataframe of scores for each tweet
        """
        if not self.bearer_token:
            warnings.warn("No bearer token provided on instantiation.")
            return {}

        scores = {}
        twitterparser = TwitterParser(self.bearer_token)
        for topic in topics:
            tweets = twitterparser.get_tweets(
                topic, start_time=start_time, end_time=end_time
            )
            scored_frame = self.sia.analyze_tweet_text(tweets)
            scores[topic] = scored_frame
        return scores

    def twitter_sentiment_summary(
        self,
        topics: list,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
    ):
        # not yet implemented
        pass


if __name__ == "__main__":
    darthvader = Isaiah(
        news_source="newsapi",
        splitting=False,
        weights={"title": 0.1, "desc": 0.1, "text": 0.8},
    )  # splitting means that it recursively splits a large text into sentences and analyzes each individually

    args = [sys.argv[1:]] if sys.argv[1:] else ["tesla"]  # default args

    scores = darthvader.news_sentiment(
        *args,
        window=3,  # how many days back from up_to to get news from
    )  # latest date to get news from

    print(scores)