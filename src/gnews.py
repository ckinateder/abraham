from newspaper import Article, ArticleException
from GoogleNews import GoogleNews
from threading import Thread
import pandas as pd
import time
from tqdm import tqdm


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

    def get_articles(self, search_term, period="2d"):
        """
        Get all articles
        """
        start = time.time()
        self.googlenews.set_period(period)  # set the range
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