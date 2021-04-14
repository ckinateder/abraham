from pandas.io import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from pprint import pprint

BASE_URL = "https://newsapi.org/v2/everything?"
API_KEY = open("keys/newsapi_org").read().strip()


def fetch_json(
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


def clean_response(jsonfile):
    """
    Cleanup the json response
    """
    results = []
    for i in range(len(jsonfile)):
        cleaned_item = {}
        cleaned_item["title"] = jsonfile[i]["title"]
        cleaned_item["author"] = jsonfile[i]["author"]
        cleaned_item["source"] = jsonfile[i]["source"]
        cleaned_item["description"] = jsonfile[i]["description"]
        cleaned_item["content"] = jsonfile[i]["content"]
        cleaned_item["publishedAt"] = jsonfile[i]["publishedAt"]
        cleaned_item["url"] = jsonfile[i]["url"]
        cleaned_item["urlToImage"] = jsonfile[i]["urlToImage"]
        results.append(cleaned_item)

    return results


def cleaned_to_df(cleaned_dict):
    """
    Take a cleaned dictionary and return a pandas dataframe
    """
    return pd.DataFrame(cleaned_dict)


def fetch(searchfor, from_date=(datetime.now() - timedelta(1)).strftime("%Y-%m-%d")):
    """
    Wrap everything to one function
    """
    jresponse = fetch_json(searchfor, from_date=from_date)
    cleaned = clean_response(jresponse)
    cleaned_df = cleaned_to_df(cleaned)
    return cleaned_df


if __name__ == "__main__":
    df = fetch("trump")
    print(df)