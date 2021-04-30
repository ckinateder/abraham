from typing import Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import nltk
from GoogleNews import GoogleNews
from newspaper import Article, ArticleException
from tqdm import tqdm, trange
from threading import Thread
from datetime import datetime, timedelta
import re
import pandas as pd
import time, requests
import warnings
import sys, logging


from dateutil.parser._parser import UnknownTimezoneWarning
import flair

logging.getLogger("flair").setLevel(logging.WARNING)

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
    get_articles
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
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
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
        start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
            how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
        end_time : str = datetime.now().strftime(TWITTER_TF)
            how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'

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
            "from": start_time,
            "to": end_time,
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
        self,
        searchfor,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
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
        cleaned_df = pd.DataFrame()
        try:
            jresponse = self.fetch_json(
                searchfor, start_time=start_time, end_time=end_time
            )
            cleaned = self.clean_response(jresponse)
            cleaned_df = self.cleaned_to_df(cleaned)
        except Exception as e:
            warnings.warn("Issue pulling from news api ... try again later.")
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
    get_articles
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
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
    ):
        """Gets articles for a single search term

        Parameters
        ----------
        searchfor: str
            term to search for
        start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
            how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
        end_time : str = datetime.now().strftime(TWITTER_TF)
            how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'

        Returns
        -------
        pandas.DataFrame
            a dataframe of the results with columns ['title', 'author', 'source', 'desc',
                                                    'text', 'datetime', 'url', 'urlToImage']
        """
        start = time.time()
        # use settimerange instead
        end_date = datetime.strptime(end_time, TWITTER_TF).strftime(GOOGLENEWS_TF)
        start_date = datetime.strptime(start_time, TWITTER_TF).strftime(GOOGLENEWS_TF)
        self.googlenews.set_time_range(start_date, end_date)
        self.googlenews.get_news(search_term)  # get the news
        results = self.googlenews.results()  # get the results

        self.pbar = tqdm(
            total=len(results),
            unit="article",
            desc=search_term,
            leave=False,
            disable=self.tqdisable,
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

    def __init__(self, bearer_token, tqdisable=False) -> None:
        """
        Parameters
        ----------
        bearer_token : str
            twitter api bearer token
        """
        self.bearer_token = bearer_token
        self.tqdisable = tqdisable

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
        pages=1,
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

        tweets = pd.DataFrame()
        for page in range(pages):
            try:
                response = requests.get(
                    TWITTER_URL,
                    params=params,
                    headers={"authorization": "Bearer " + self.bearer_token},
                )
                jresponse = response.json()
                if response.status_code == 200:
                    for tweet in tqdm(
                        jresponse["data"],
                        leave=False,
                        desc=f"{topic} tweets",
                        dynamic_ncols=True,
                        disable=self.tqdisable,
                    ):
                        try:
                            row = self.parse_tweet(tweet)
                            tweets = tweets.append(row, ignore_index=True)
                        except Exception as e:
                            warnings.warn(
                                f"Error while parsing tweet ... skipping ({e})"
                            )
                    params["next_token"] = jresponse["meta"]["next_token"]
                else:
                    warnings.warn(
                        f"Response code {response.status_code} recieved from twitter. Did you authenticate correctly?"
                    )
            except Exception as e:
                warnings.warn(f"Error while getting tweets ({e})")
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

    def __init__(self, tqdisable=False) -> None:
        self.vader = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()

        self.sunflair = flair.models.TextClassifier.load("en-sentiment")
        # nltk.download("vader_lexicon")
        self.tqdisable = tqdisable

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
        try:
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
            lemmatized_words = [
                self.lemmatizer.lemmatize(word) for word in filtered_words
            ]
            new_sentence = " ".join(lemmatized_words)
            return new_sentence
        except:
            return sentence

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
            frame.index,
            leave=False,
            unit="text",
            desc=f"analyze {section}",
            disable=self.tqdisable,
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

    def analyze_flair_text(self, tweets: pd.DataFrame, section: str):
        """Takes a dataframe of tweets and analyzes and saves the score for each row
        ...

        Parameters
        ----------
        tweet_frame: pd.DataFrame
            dataframe containg section
        section: pd.DataFrame
            section of tweet_frame

        Returns
        -------
        pandas.DataFrame
            new dataframe of scores
        """
        # we will append probability and sentiment preds later
        probs = []
        sentiments = []
        newframe = tweets.copy()
        try:
            newframe[section] = newframe[section].apply(self.normalize_text)

            for tweet in tqdm(
                newframe[section].to_list(),
                desc=section,
                leave=False,
                dynamic_ncols=True,
                disable=self.tqdisable,
            ):
                # if tweet empty
                if not tweet:
                    tweet = ""  # make string if none
                if len(tweet) == 0:
                    probs.append("NEUTRAL")
                    sentiments.append("NEUTRAL")
                else:
                    # make prediction
                    sentence = flair.data.Sentence(tweet)
                    self.sunflair.predict(sentence)
                    # extract sentiment prediction
                    probs.append(sentence.labels[0].score)  # numerical score 0-1
                    sentiments.append(
                        sentence.labels[0].value
                    )  # 'POSITIVE' or 'NEGATIVE'
        except:
            probs = ["NEUTRAL"] * newframe.shape[0]
            sentiments = ["NEUTRAL"] * newframe.shape[0]
            # add probability and sentiment predictions to tweets dataframe
        newframe["probability"] = probs
        newframe["sentiment"] = sentiments
        newframe = newframe[newframe.probability != "NEUTRAL"]
        newframe = newframe[newframe.sentiment != "NEUTRAL"]
        return newframe


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
    tqdisable : bool
        disale progressbars

    Methods
    -------
    get_articles
        gets articles for a single search term
    twitter_sentiment
        takes a list of topics and gets the raw scores for each
        (per topic per text type per row)
    twitter_summary
        takes a list of topics and computes the avg scores for each
    news_summary
        takes a list of topics and computes the avg scores for each
    news_sentiment
        takes a list of topics and gets the raw scores for each
        (per topic per text type per row)
    summary
        takes a list of topics and gets the news and twitter summary
        avg for each topic
    news_summary_interval
        takes a list of topics and gets the NEWS summary over each
        'period' intervals from oldest to newest
    twitter_summary_interval
        takes a list of topics and gets the TWITTER summary over each
        'period' intervals from oldest to newest
    summary_interval
        takes a list of topics and gets the NEWS+TWITTER AVG
        summary over each 'period' intervals from oldest to newest
    """

    def __init__(
        self,
        news_source="google",
        newsapi_key=None,
        bearer_token=None,
        weights={"title": 0.33, "desc": 0.33, "text": 0.34},
        loud=False,
        tqdisable=False,
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
        weights : dict = {"title": 0.33, "desc": 0.33, "text": 0.34}
            how to weight the title, desc, and text attributes
        loud : dict = False
            print unnecessary output (for debugging ususally)
        tqdisable : bool = False
            disale progressbars
        """
        self.sia = Elijiah(tqdisable=True)
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
        self.weights = weights
        self.loud = loud
        self.bearer_token = bearer_token
        self.tqdisable = tqdisable

    def get_articles(
        self,
        topics: list,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
    ) -> Dict:
        """Takes a list of topics and returns a dict of topics : pd.dataframe

        Parameters
        ----------
        topics : list
            list of terms to search for
        start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
            how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
        end_time : str = datetime.now().strftime(TWITTER_TF)
            how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'

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
                topic, start_time=start_time, end_time=end_time
            )
        return topic_results

    def news_summary(
        self,
        topics: list,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
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
            a dict of dicts arranged as {topic: scores},
            where scores is a tuple (positive count, negative cound)
        """
        scores = {}
        raws = self.news_sentiment(topics, start_time=start_time, end_time=end_time)
        for topic in raws:
            title = (
                raws[topic]["title"]
                .loc[raws[topic]["title"]["sentiment"] == "POSITIVE"]
                .dropna()
                .shape[0],
                raws[topic]["title"]
                .loc[raws[topic]["title"]["sentiment"] == "NEGATIVE"]
                .dropna()
                .shape[0],
            )
            desc = (
                raws[topic]["desc"]
                .loc[raws[topic]["desc"]["sentiment"] == "POSITIVE"]
                .dropna()
                .shape[0],
                raws[topic]["desc"]
                .loc[raws[topic]["desc"]["sentiment"] == "NEGATIVE"]
                .dropna()
                .shape[0],
            )
            text = (
                raws[topic]["text"]
                .loc[raws[topic]["text"]["sentiment"] == "POSITIVE"]
                .dropna()
                .shape[0],
                raws[topic]["text"]
                .loc[raws[topic]["text"]["sentiment"] == "NEGATIVE"]
                .dropna()
                .shape[0],
            )

            # apply weights here
            total = (
                round(
                    title[0]
                    * self.weights["title"]
                    * (100 / raws[topic]["title"].shape[0])
                    + desc[0]
                    * self.weights["desc"]
                    * (100 / raws[topic]["desc"].shape[0])
                    + text[0]
                    * self.weights["text"]
                    * (100 / raws[topic]["text"].shape[0]),
                    1,
                ),
                round(
                    title[1]
                    * self.weights["title"]
                    * (100 / raws[topic]["title"].shape[0])
                    + desc[1]
                    * self.weights["desc"]
                    * (100 / raws[topic]["desc"].shape[0])
                    + text[1]
                    * self.weights["text"]
                    * (100 / raws[topic]["text"].shape[0]),
                    1,
                ),
            )
            scores[topic] = total
        return scores

    def news_sentiment(
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
        start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
            how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
        end_time : str = datetime.now().strftime(TWITTER_TF)
            how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'

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

        articles = self.get_articles(topics, start_time=start_time, end_time=end_time)
        scores = {}
        for topic in articles:
            titles = self.sia.analyze_flair_text(
                articles[topic][["title", "datetime"]], "title"
            )
            desc = self.sia.analyze_flair_text(
                articles[topic][["desc", "datetime"]], "desc"
            )
            text = self.sia.analyze_flair_text(
                articles[topic][["text", "datetime"]], "text"
            )
            scores[topic] = {"title": titles, "desc": desc, "text": text}
        return scores

    def twitter_summary(
        self,
        topics: list,
        size: int = 100,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
    ):
        """Gets the summary sentiment for each topic from twitter

        Parameters
        ----------
        topics : list
            list of terms to search for
        start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
            how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
        end_time : str = datetime.now().strftime(TWITTER_TF)
            how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'

        Returns
        -------
        scores : dict
            a dict of dicts arranged as {topic: scores},
            where scores is a tuple (positive count, negative cound)
        """
        scores = {}
        raws = self.twitter_sentiment(
            topics, size=size, start_time=start_time, end_time=end_time
        )
        for topic in raws:
            try:
                scores[topic] = (
                    round(
                        raws[topic]
                        .loc[raws[topic]["sentiment"] == "POSITIVE"]
                        .dropna()
                        .shape[0]
                        * (100 / raws[topic].shape[0]),
                        1,
                    ),
                    round(
                        raws[topic]
                        .loc[raws[topic]["sentiment"] == "NEGATIVE"]
                        .dropna()
                        .shape[0]
                        * (100 / raws[topic].shape[0]),
                        1,
                    ),
                )
            except:
                scores[topic] = (-1, -1)
        return scores

    def twitter_sentiment(
        self,
        topics: list,
        size=100,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
    ):
        """Gets the WHOLE sentiment for each topic from twitter. No or minimal averaging occurs.

        Parameters
        ----------
        topics : list
            list of terms to search for
        size : int = 100
            roughly how many tweets to get
        start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
            how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
        end_time : str = datetime.now().strftime(TWITTER_TF)
            how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'

        Returns
        -------
        scores : dict
            a dict of dataframe of scores for each topic
        """
        if not self.bearer_token:
            warnings.warn("No bearer token provided on instantiation.")
            return {}

        scores = {}
        twitterparser = TwitterParser(self.bearer_token, tqdisable=True)
        for topic in topics:
            tweets = twitterparser.get_tweets(
                topic, pages=int(size / 100), start_time=start_time, end_time=end_time
            )
            scored_frame = self.sia.analyze_flair_text(tweets, "text")
            scores[topic] = scored_frame
        return scores

    def summary(
        self,
        topics: list,
        start_time=(datetime.now() - timedelta(2)).strftime(TWITTER_TF),
        end_time=datetime.now().strftime(TWITTER_TF),
        weights={"news": 0.5, "twitter": 0.5},
    ):
        """Gets the WHOLE sentiment from news and twitter for each topic.

        Parameters
        ----------
        topics : list
            list of terms to search for
        start_time : str = (datetime.now() - timedelta(2)).strftime(TWITTER_TF)
            how far back to search from in time format %Y-%m-%dT%H:%M:%SZ'
        end_time : str = datetime.now().strftime(TWITTER_TF)
            how recent to search from in time format %Y-%m-%dT%H:%M:%SZ'
        weights : dict = {"news": 0.5, "twitter": 0.5}
            how to weight the news results to the twitter results

        Returns
        -------
        total : dict
            a dict of dataframe of scores for each topic
        """
        twitter = self.twitter_summary(
            topics=topics, start_time=start_time, end_time=end_time
        )
        news = self.news_summary(
            topics=topics, start_time=start_time, end_time=end_time
        )
        total = {}
        for topic in twitter:
            try:
                total[topic] = (
                    round(
                        (
                            twitter[topic][0] * weights["twitter"]
                            + news[topic][0] * weights["news"]
                        ),
                        1,
                    ),
                    round(
                        (
                            twitter[topic][1] * weights["twitter"]
                            + news[topic][1] * weights["news"]
                        ),
                        1,
                    ),
                )
            except Exception as e:
                warnings.warn(f"Error getting total for {topic} ({e})")
        return total

        ## now for time travel

    def _intervals(self, start, end, delta):
        """Get the number of delta-size intervals in start to end

        Parameters
        ----------
        start : datetime.datetime
            the start date
        end : datetime.datetime
            the end date
        delta : datetime.timedelta
            the interval size

        Returns
        -------
        count : int
            number of delta-size intervals in start-end
        """
        curr = start
        count = 0
        while curr < end:
            count += 1
            curr += delta
        return count

    def _is_bday(self, dt):
        """Returns true if dt between 9:30 and 4:30"""
        return bool(len(pd.bdate_range(dt, dt)))

    def news_summary_interval(
        self,
        topics,
        oldest=datetime.now() - timedelta(days=1),
        newest=datetime.now(),
        period=timedelta(hours=1),
    ):
        """Get the NEWS summary over each 'period' intervals from oldest to newest

        Parameters
        ----------
        topics : list
            list of topics to search for
        oldest : datetime.datetime
            oldest datetime to search from
        newest : datetime.datetime
            newest datetime to search up to
        period : timedelta
            interval to advance through with

        Returns
        -------
        results : dict
            a dictionary of results in form `topic : results dataframe`
        """
        results = {}

        for topic in topics:
            now = newest
            df = pd.DataFrame(columns=["timestamp", "positive", "negative"])

            for i in trange(
                self._intervals(oldest, newest, period),
                leave=True,
                dynamic_ncols=True,
                desc="backtest",
                disable=self.tqdisable,
            ):
                pre = now - period

                scores = self.news_summary(
                    [topic],
                    start_time=pre.strftime(TWITTER_TF),
                    end_time=now.strftime(TWITTER_TF),
                )[topic]

                # add to dataframe
                df = df.append(
                    {
                        "timestamp": now,
                        "positive": scores[0],
                        "negative": scores[1],
                        "lag": period,
                    },
                    ignore_index=True,
                )
                # move back
                now = pre
            results[topic] = df
        return results

    def twitter_summary_interval(
        self,
        topics,
        oldest=datetime.now() - timedelta(days=1),
        newest=datetime.now(),
        period=timedelta(hours=1),
    ):
        """Get the TWITTER summary over each 'period' intervals from oldest to newest

        Parameters
        ----------
        topics : list
            list of topics to search for
        oldest : datetime.datetime
            oldest datetime to search from
        newest : datetime.datetime
            newest datetime to search up to
        period : timedelta
            interval to advance through with

        Returns
        -------
        results : dict
            a dictionary of results in form `topic : results dataframe`
        """
        results = {}

        for topic in topics:
            now = newest
            df = pd.DataFrame(columns=["timestamp", "positive", "negative"])

            for i in trange(
                self._intervals(oldest, newest, period),
                leave=True,
                dynamic_ncols=True,
                desc="backtest",
                disable=self.tqdisable,
            ):
                pre = now - period

                scores = self.twitter_summary(
                    [topic],
                    start_time=pre.strftime(TWITTER_TF),
                    end_time=now.strftime(TWITTER_TF),
                )[topic]

                # add to dataframe
                df = df.append(
                    {
                        "timestamp": now,
                        "positive": scores[0],
                        "negative": scores[1],
                        "lag": period,
                    },
                    ignore_index=True,
                )
                # move back
                now = pre
            results[topic] = df
        return results

    def summary_interval(
        self,
        topics,
        oldest=datetime.now() - timedelta(days=1),
        newest=datetime.now(),
        period=timedelta(hours=1),
    ):
        """Get the TOTAL summary (twitter/newsapi) over each 'period' intervals from oldest to newest

        Parameters
        ----------
        topics : list
            list of topics to search for
        oldest : datetime.datetime
            oldest datetime to search from
        newest : datetime.datetime
            newest datetime to search up to
        period : timedelta
            interval to advance through with

        Returns
        -------
        results : dict
            a dictionary of results in form `topic : results dataframe`
        """
        results = {}

        for topic in topics:
            now = newest
            df = pd.DataFrame(columns=["timestamp", "positive", "negative"])

            for i in trange(
                self._intervals(oldest, newest, period),
                leave=True,
                dynamic_ncols=True,
                desc="backtest",
                disable=self.tqdisable,
            ):
                pre = now - period

                scores = self.summary(
                    [topic],
                    start_time=pre.strftime(TWITTER_TF),
                    end_time=now.strftime(TWITTER_TF),
                )[topic]

                # add to dataframe
                df = df.append(
                    {
                        "timestamp": now,
                        "positive": scores[0],
                        "negative": scores[1],
                        "lag": period,
                    },
                    ignore_index=True,
                )
                # move back
                now = pre
            results[topic] = df
        return results