from json import JSONDecodeError

import feedparser
import google.cloud.bigquery
import pandas as pd
import requests
from google.api_core.exceptions import GoogleAPICallError
from iexfinance.stocks import Stock

from ..spreadsheet_error import VALUE_ERROR


def json(url_or_str: str) -> pd.DataFrame:
    """Import a JSON file from the web.

    Example:
        =data.json("https://data.cityofnewyork.us/resource/erm2-nwe9.json")
    """
    try:
        return pd.read_json(url_or_str)
    except JSONDecodeError as e:
        return VALUE_ERROR.with_message(str(e))


def geojson(url_or_str: str) -> pd.DataFrame:
    """Import a GeoJSON file from the web.

    Example:
        =data.geojson("https://data.cityofnewyork.us/resource/erm2-nwe9.geojson")
    """
    import geo

    return geo.dataset(url_or_str)


def csv(url_or_str: str) -> pd.DataFrame:
    """Import a CSV file from the web.

    Example:
        =data.csv("https://data.cityofnewyork.us/resource/erm2-nwe9.csv")
    """
    return pd.read_csv(url_or_str)


def rss(url: str) -> pd.DataFrame:
    """Import an RSS feed from the web.

    Example:
        =data.rss("https://www.reddit.com/r/Python/.rss")
    """
    feed = feedparser.parse(url)
    entries = feed.entries
    data = [
        {
            key: entry[key]
            for key in entry
            if key not in ["published_parsed", "updated_parsed"]
            and isinstance(entry[key], str)
        }
        for entry in entries
    ]

    return pd.DataFrame(data)


def web_table(url: str, idx=-1) -> pd.DataFrame:
    """Import a table from the web.

    url: the url of the table
    idx: the index of the table on the page. Defaults to the largest table.

    Example:
        =data.web_table("https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)")
    """

    tables = pd.read_html(url)
    if idx == -1:
        return max(tables, key=lambda t: t.shape[0])
    if idx >= len(tables):
        return VALUE_ERROR.with_message(
            f"Table index {idx} out of range - only {len(tables)} tables found"
        )
    return tables[idx]


def big_query(sql: str) -> pd.DataFrame:
    """Query the Google BigQuery database.

    sql: the query to run

    Example:
        =data.big_query("SELECT * FROM `bigquery-public-data.covid19_open_data.covid19_open_data` LIMIT 10")
    """
    client = google.cloud.bigquery.Client()
    try:
        return client.query(sql).to_dataframe(geography_as_object=True)
    except GoogleAPICallError as e:
        return VALUE_ERROR.with_message(str(e))


def web_search(query: str, count=10) -> pd.DataFrame:
    """Search the web and return the top results.

    query: the search query

    Example:
        =data.web_search("python pandas")
    """
    import requests

    res = requests.get(
        "https://api.bing.microsoft.com/v7.0/search",
        params={"q": query},
    )
    res.raise_for_status()
    columns_to_keep = [
        "name",
        "url",
        "snippet",
        "cachedPageUrl",
        "language",
        "thumbnailUrl",
        "datePublished",
    ]
    df = pd.DataFrame(res.json()["webPages"]["value"])[columns_to_keep]
    return df.fillna("")


def stock_lookup(symbols: str | list[str]) -> pd.DataFrame:
    """Lookup stock information for one or more stocks.

    symbols: the stock symbols to lookup

    Example:
        =data.stock_lookup("AAPL")
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    stocks = Stock([*symbols])
    data = stocks.get_quote()
    columns_to_keep = [
        "avgTotalVolume",
        "close",
        "high",
        "latestPrice",
        "low",
        "marketCap",
        "open",
        "peRatio",
        "previousClose",
        "week52High",
        "week52Low",
    ]
    df = pd.DataFrame(data)
    df["open"] = df["iexOpen"]
    df["close"] = df["iexClose"]
    return df[columns_to_keep]


def geocode(address: str) -> tuple[float, float]:
    """Get the latitude and longitude of an address.

    address: the address to geocode

    Example:
        =data.geocode("1600 Amphitheatre Parkway, Mountain View, CA")
    """
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if data["status"] != "OK":
        return VALUE_ERROR.with_message(data["status"])

    results = data["results"]
    if not results:
        return VALUE_ERROR.with_message("No results found")
    location = results[0]["geometry"]["location"]
    lat = location["lat"]
    lng = location["lng"]
    return lat, lng


__all__ = [
    "big_query",
    "csv",
    "geocode",
    "geojson",
    "json",
    "rss",
    "stock_lookup",
    "web_search",
    "web_table",
]
