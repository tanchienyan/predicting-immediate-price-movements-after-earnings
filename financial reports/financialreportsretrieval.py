import json
import pandas as pd

# Load the JSON data from the file
with open("company_tickers_exchange.json", "r") as f:
    CIK_dict = json.load(f)

# Dataset contains 2 sections
# CIK_dict.keys()
# CIK_dict["fields"]
# Out: ['cik', 'name', 'ticker', 'exchange']
# print("Number of company records:", len(CIK_dict["data"]))
# Number of company records: 10765
CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])

# Get ticker symbol from user
ticker = input("Ticker Symbol: ").upper()
# Find company row with given ticker
CIK_df[CIK_df["ticker"] == ticker]
CIK = CIK_df[CIK_df["ticker"] == ticker].cik.values[0]

# Finding companies of same industry:
# Example:
# substring = "oil"
# CIK_df[CIK_df["name"].str.contains(substring, case=False)]

# preparation of input data, using ticker and CIK set earlier
url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"

import requests
header = {
  "User-Agent": "chienyantan@gmail.com"#, # remaining fields are optional
#    "Accept-Encoding": "gzip, deflate",
#    "Host": "data.sec.gov"
}

company_filings = requests.get(url, headers=header).json()
# company_filings["addresses"]
company_filings_df = pd.DataFrame(company_filings["filings"]["recent"])

# Filter only Annual reports
company_filings_df[company_filings_df.form == "10-K"]
access_number = company_filings_df[company_filings_df.form == "10-K"].accessionNumber.values[0].replace("-", "")
file_name = company_filings_df[company_filings_df.form == "10-K"].primaryDocument.values[0]
url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{file_name}"

# Dowloading and saving requested document to working directory
req_content = requests.get(url, headers=header).content.decode("utf-8")

with open(file_name, "w") as f:
    f.write(req_content)

from weasyprint import HTML

