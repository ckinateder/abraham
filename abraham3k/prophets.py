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

warnings.simplefilter("ignore", UnknownTimezoneWarning)

# define time formats for each
GOOGLENEWS_TF = "%m/%d/%Y"
NEWSAPI_TF = "%Y-%m-%d"

BASE_URL = "https://newsapi.org/v2/everything?"


class NewsAPI:
    def __init__(
        self,
        api_key,
    ) -> None:
        self.api_key = api_key

    def fetch_json(
        self,
        searchfor,
        url=BASE_URL,
        pagesize=100,
        page=1,
        language="en",
        from_date=(datetime.now() - timedelta(7)).strftime(NEWSAPI_TF),
        to_date=datetime.now().strftime(NEWSAPI_TF),
    ):
        """
        Search the news for a search term.
        """
        params = {
            "q": searchfor,
            "pageSize": pagesize,
            "apiKey": self.api_key,
            "language": language,
            "page": page,
            "from": from_date,
            "to": to_date,
        }
        response = requests.get(url, params=params)
        json_response = response.json()["articles"]
        return json_response

    def clean_response(self, jsonfile):
        """
        Cleanup the json response
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
        """
        Take a cleaned dictionary and return a pandas dataframe
        """
        return pd.DataFrame(cleaned_dict)

    def get_articles(
        self, searchfor, up_to=datetime.now().strftime(NEWSAPI_TF), window=1
    ):
        """
        Wrap everything to one function
        """
        period = (datetime.now() - timedelta(window)).strftime(NEWSAPI_TF)
        jresponse = self.fetch_json(searchfor, from_date=period, to_date=up_to)
        cleaned = self.clean_response(jresponse)
        cleaned_df = self.cleaned_to_df(cleaned)
        return cleaned_df


class GoogleNewsParser:
    def __init__(self) -> None:
        self.googlenews = GoogleNews()  # create news object

    def _get_text(
        self, inst
    ):  # download the article text for each link and save as a string
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
        up_to=datetime.now().strftime(NEWSAPI_TF),
        window=2,  # how many days back to go
    ):
        """
        Get all articles
        """
        start = time.time()
        # use settimerange instead
        end_date = datetime.strptime(up_to, NEWSAPI_TF).strftime(GOOGLENEWS_TF)
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


class Elijiah:
    """
    Perform sentiment analysis on sentences
    """

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()

    def sentiment_analyzer_sent(self, sentence: str):
        score = self.analyzer.polarity_scores(sentence)
        print("{:-<40} {}".format(sentence, str(score)))

    def normalize_text(self, sentence: str):
        # normalize the sentence
        subbed = sentence
        clean = re.compile("<.*?>")
        subbed = re.sub(clean, "", subbed)
        subbed = re.sub("\[+(.*?)chars\]", "", subbed)
        word_list = subbed.split()
        filtered_words = [
            word for word in word_list if word not in stopwords.words("english")
        ]
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
        new_sentence = " ".join(lemmatized_words)
        return new_sentence

    def analyze_texts(self, frame: pd.DataFrame, section: str, recursive: bool = False):
        """
        Takes a list of sentences and returns a dataframe the sentences and scores
        Example -
            neg    neu    pos  compound                   sentence
        0  0.123  0.877  0.000   -0.1027   Cramer call Coinbase 'real deal,' warns invest
        ...
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
                    score = self.analyzer.polarity_scores(sentence)
                    if score["compound"] != 0:  # remove all zeroes
                        score["sentence"] = sentence
                        score["datetime"] = date
                        scores.append(score)
            else:
                sentence = self.normalize_text(item)
                score = self.analyzer.polarity_scores(sentence)
                if score["compound"] != 0:  # remove all zeroes
                    score["sentence"] = sentence
                    score["datetime"] = date
                    scores.append(score)
        scores = pd.DataFrame(scores)
        return scores


class Isaiah:
    """
    Wraps everything into getting the sentiment for a given topic
    """

    def __init__(
        self,
        news_source="google",
        api_key=None,
        splitting=False,
        weights={"title": 0.2, "desc": 0.3, "text": 0.5},
        loud=False,
    ) -> None:
        self.sia = Elijiah()
        if news_source == "newsapi":
            if api_key:
                self.news_source = "newsapi"
                self.newsparser = NewsAPI(api_key=api_key)
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

    def get_articles(
        self,
        topics: list,
        window: int = 2,
        up_to: str = datetime.now().strftime(NEWSAPI_TF),
    ) -> Dict:
        """
        Takes a list of topics and returns a dict
        ex -
        {
            'coinbase': <pd.DataFrame>,
        }
        """
        topic_results = {}
        for topic in topics:
            topic_results[topic] = self.newsparser.get_articles(
                topic, window=window, up_to=up_to
            )
        return topic_results

    def compute_total_avg(self, results_df: pd.DataFrame, meta: dict):
        # compute the average for each column
        # titles = list(results_df.title)
        # desc = list(results_df.desc)
        # text = list(results_df.text)
        title_avg = round(
            self.sia.analyze_texts(
                results_df, "title", recursive=self.splitting
            ).compound.mean(),
            8,
        )
        desc_avg = round(
            self.sia.analyze_texts(
                results_df, "desc", recursive=self.splitting
            ).compound.mean(),
            8,
        )
        text_avg = round(
            self.sia.analyze_texts(
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
        # test all and build a dict
        scores = {}
        for topic in topic_results:
            classification = self.compute_total_avg(topic_results[topic], meta=meta)
            scores[topic] = classification
        return scores

    def sentiment_summary(
        self,
        topics: list,
        window: int = 2,
        up_to: str = datetime.now().strftime(NEWSAPI_TF),
    ):
        """
        Main function
        """
        articles = self.get_articles(topics, window=window, up_to=up_to)
        scores = self.score_all(articles, meta={"window": window, "up_to": up_to})
        return scores

    def sentiment(
        self,
        topics: list,
        window: int = 2,
        up_to: str = datetime.now().strftime(NEWSAPI_TF),
    ):
        """
        Returns the indepth scores as a two-deep dict.
        """
        articles = self.get_articles(topics, window=window, up_to=up_to)

        scores = {}
        for topic in articles:
            titles = self.sia.analyze_texts(
                articles[topic], "title", recursive=self.splitting
            )
            desc = self.sia.analyze_texts(
                articles[topic], "desc", recursive=self.splitting
            )
            text = self.sia.analyze_texts(
                articles[topic], "text", recursive=self.splitting
            )
            scores[topic] = {"title": titles, "desc": desc, "text": text}
        return scores


if __name__ == "__main__":
    # print(f"Abraham Version: {open('version').read().strip()}")
    darthvader = Isaiah(
        news_source="newsapi",
        splitting=False,
        weights={"title": 0.1, "desc": 0.1, "text": 0.8},
    )  # splitting means that it recursively splits a large text into sentences and analyzes each individually

    args = [sys.argv[1:]] if sys.argv[1:] else ["tesla"]  # default args

    scores = darthvader.sentiment(
        *args,
        window=3,  # how many days back from up_to to get news from
    )  # latest date to get news from

    print(scores)