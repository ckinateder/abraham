from typing import Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from gnews import NewsParser
from newsapi import NewsAPI
from tqdm import tqdm
import re
import pandas as pd
import statistics


class Anakin:
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


class Darth:
    """
    Wraps everything into getting the sentiment for a given topic
    """

    def __init__(self, period="1d", news_source="google", splitting=False) -> None:
        self.sia = Anakin()
        if news_source == "newsapi":
            self.newsparser = NewsAPI()
        else:
            self.newsparser = NewsParser()
        self.period = period  # date range back to search for
        self.splitting = splitting  # does sia.analyze use recursion to split?

    def get_articles(self, topics: list) -> Dict:
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
                topic, period=self.period
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

    def score_all(self, topic_results: list):
        # test all and build a dict
        scores = {}
        for topic in topic_results:
            total_avg, classified = self.compute_total_avg(topic_results[topic])
            scores[topic] = {"avg": total_avg, "nice": classified}
        return scores

    def sentiment(self, topics):
        """
        Main function
        """
        articles = self.get_articles(topics)
        scores = self.score_all(articles)
        return scores