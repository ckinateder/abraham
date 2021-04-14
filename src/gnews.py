from newspaper import Article, ArticleException
from GoogleNews import GoogleNews
from concurrent.futures import ThreadPoolExecutor, as_completed


class NewsParser:
    def __init__(self) -> None:
        self.googlenews = GoogleNews()  # create news object

    def get_text(
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
            print(
                f"Failed on {inst['link'].replace('news.google.com/./articles/', '')[:25]}..."
            )
            inst["text"] = ""
        return inst["text"]

    def get_articles(self, search_term):
        self.googlenews.get_news(search_term)  # get the news
        results = self.googlenews.results()  # get the results

        processes = []  # multi thread the execution
        with ThreadPoolExecutor() as executor:
            for i in results:
                processes.append(executor.submit(self.get_text, i))
        return results