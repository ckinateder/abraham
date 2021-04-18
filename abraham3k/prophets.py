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
import statistics
import time, requests

WINDOW_TF = "%m/%d/%Y"


BASE_URL = "https://newsapi.org/v2/everything?"

try:
    API_KEY = open("keys/newsapi_org").read().strip()
except FileNotFoundError as e:
    print(
        "Couldn't load API key for newsapi.org (No such file or directory: 'keys/newsapi_org)"
    )
    API_KEY = ""
except Exception as e:
    print(f"Couldn't load API key for newsapi.org ({e})")
    API_KEY = ""


class NewsAPI:
    def __init__(self) -> None:
        pass

    def fetch_json(
        self,
        searchfor,
        url=BASE_URL,
        api_key=API_KEY,
        pagesize=100,
        page=1,
        language="en",
        from_date=(datetime.now() - timedelta(7)).strftime("%Y-%m-%d"),
    ):
        """
        Search the news for a search term.
        """
        params = {
            "q": searchfor,
            "pageSize": pagesize,
            "apiKey": api_key,
            "language": language,
            "page": page,
            "from": from_date,
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
            cleaned_item["publishedAt"] = jsonfile[i]["publishedAt"]
            cleaned_item["url"] = jsonfile[i]["url"]
            cleaned_item["urlToImage"] = jsonfile[i]["urlToImage"]
            results.append(cleaned_item)

        return results

    def cleaned_to_df(self, cleaned_dict):
        """
        Take a cleaned dictionary and return a pandas dataframe
        """
        return pd.DataFrame(cleaned_dict)

    def get_articles(self, searchfor, period="1d"):
        """
        Wrap everything to one function
        """
        period = (datetime.now() - timedelta(int(period.replace("d", "")))).strftime(
            "%Y-%m-%d"
        )
        jresponse = self.fetch_json(searchfor, from_date=period)
        cleaned = self.clean_response(jresponse)
        cleaned_df = self.cleaned_to_df(cleaned)
        return cleaned_df


class NewsParser:
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
        up_to=datetime.now().strftime(WINDOW_TF),
        window=2,  # how many days back to go
    ):
        """
        Get all articles
        """
        start = time.time()
        # use settimerange instead
        end_date = up_to
        start_date = (datetime.now() - timedelta(window)).strftime(WINDOW_TF)
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

    def analyze_texts(self, sentences: list, recursive: bool = False):
        """
        Takes a list of sentences and returns a dataframe the sentences and scores
        Example -
            neg    neu    pos  compound                   sentence
        0  0.123  0.877  0.000   -0.1027   Cramer call Coinbase 'real deal,' warns invest
        ...
        """
        scores = []
        for item in tqdm(sentences, leave=False, unit="text", desc="analyze"):
            # iterate through list of sent tokenize
            if recursive:
                for sentence in sent_tokenize(item):
                    sentence = self.normalize_text(sentence)
                    score = self.analyzer.polarity_scores(sentence)
                    if score["compound"] != 0:  # remove all zeroes
                        score["sentence"] = sentence
                        scores.append(score)
            else:
                sentence = self.normalize_text(item)
                score = self.analyzer.polarity_scores(sentence)
                if score["compound"] != 0:  # remove all zeroes
                    score["sentence"] = sentence
                    scores.append(score)
        scores = pd.DataFrame(scores)
        return scores


class Isaiah:
    """
    Wraps everything into getting the sentiment for a given topic
    """

    def __init__(self, news_source="google", splitting=False) -> None:
        self.sia = Elijiah()
        if news_source == "newsapi":
            self.newsparser = NewsAPI()
        else:
            self.newsparser = NewsParser()
        self.splitting = splitting  # does sia.analyze use recursion to split?

    def get_articles(
        self,
        topics: list,
        window: int = 2,
        up_to: str = datetime.now().strftime(WINDOW_TF),
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

    def compute_total_avg(self, results_df: pd.DataFrame, loud=False):
        # compute the average for each column
        titles = list(results_df.title)
        desc = list(results_df.desc)
        text = list(results_df.text)
        title_avg = self.sia.analyze_texts(
            titles, recursive=self.splitting
        ).compound.mean()
        desc_avg = self.sia.analyze_texts(
            desc, recursive=self.splitting
        ).compound.mean()
        text_avg = self.sia.analyze_texts(
            text, recursive=self.splitting
        ).compound.mean()
        total_avg = statistics.mean([title_avg, desc_avg, text_avg])
        # classify
        if total_avg >= 0.05:
            classified = "positive"
        elif total_avg <= -0.05:
            classified = "negative"
        else:
            classified = "neutral"

        if loud:
            print(f"Title avg: {round(title_avg*100,2)}% (compound={title_avg:.8f})")
            print(f"Desc avg: {round(desc_avg*100,2)}% (compound={desc_avg:.8f})")
            print(f"Text avg: {round(text_avg*100,2)}% (compound={text_avg:.8f})")
            print(
                f"--\nTotal avg: {round(total_avg*100,2)}% (compound={total_avg:.8f})"
            )
            print(f"{classified}")
        return total_avg, classified

    def score_all(self, topic_results: dict):
        # test all and build a dict
        scores = {}
        for topic in topic_results:
            total_avg, classified = self.compute_total_avg(topic_results[topic])
            scores[topic] = {"avg": total_avg, "nice": classified}
        return scores

    def sentiment(
        self,
        topics: list,
        window: int = 2,
        up_to: str = datetime.now().strftime(WINDOW_TF),
    ):
        """
        Main function
        """
        articles = self.get_articles(topics, window=window, up_to=up_to)
        scores = self.score_all(articles)
        return scores


if __name__ == "__main__":
    darthvader = Isaiah(
        news_source="google", splitting=True
    )  # splitting means that it recursively splits a large text into sentences and analyzes each individually

    # this command takes a bit of time to run because it has to download lots of articles
    scores = darthvader.sentiment(
        ["bitcoin", "biden"],
        window=1,  # how many days back from up_to to get news from
        up_to="04/18/2021",
    )  # latest date to get news from

    print(scores)