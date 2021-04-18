from newspaper import Article, ArticleException
from GoogleNews import GoogleNews
from threading import Thread
from datetime import datetime, timedelta
import pandas as pd
import time
from tqdm import tqdm

WINDOW_TF = "%m/%d/%Y"


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


if __name__ == "__main__":
    n = NewsParser()
    print(n.get_articles("tesla"))
