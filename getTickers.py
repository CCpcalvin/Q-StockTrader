# Get the tickers from nasdaq
# Here we follows the approach on https://github.com/shilewenuw/get_all_tickers/blob/master/get_all_tickers/get_tickers.py

import requests
import pandas as pd


headers = {
    'authority': 'api.nasdaq.com',
    'accept': 'application/json, text/plain, */*',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
    'origin': 'https://www.nasdaq.com',
    'sec-fetch-site': 'same-site',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'referer': 'https://www.nasdaq.com/',
    'accept-language': 'en-US,en;q=0.9',
}


def params(exchange):
    return (
        ('letter', '0'),
        ('exchange', exchange),
        ('download', 'true'),
    )


def exchange2df(exchange):
    r = requests.get('https://api.nasdaq.com/api/screener/stocks', headers=headers, params=params(exchange))
    data = r.json()['data']
    df = pd.DataFrame(data['rows'], columns=data['headers'])
    return df


def main():
    df = exchange2df("nasdaq")
    df.to_csv("tickers.csv", index=False)
    

if __name__ == "__main__":
    main()

