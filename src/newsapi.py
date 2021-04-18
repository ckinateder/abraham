from os import execve
from pandas.io import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pprint import pprint

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
