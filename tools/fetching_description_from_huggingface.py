import requests
from bs4 import BeautifulSoup
import re

def fetching_description_from_huggingface(url, timeout = 30):
  headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "vi",
    "Content-Type": "application/x-www-form-urlencoded",
    "Host": "huggingface.co",
    "Origin": "https://huggingface.co/",
    "Referer": "https://huggingface.co/",
  }
  try:
    response = requests.get(url, headers = headers, timeout = timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find(class_ = "model-card-content")
    if not content:
        return "Model description not found."
    raw_text = content.get_text()
    processed_raw_text = raw_text.replace("\t", "")
    processed_text = re.sub(r'\n+', '\n', processed_raw_text)
    return processed_text
  except requests.exceptions.Timeout:
    raise TimeoutError(f"Request to {url} timed out after {timeout} seconds.")
  except requests.exceptions.RequestException as e:
    raise ConnectionError(f"Failed to fetch from {url}: {str(e)}")