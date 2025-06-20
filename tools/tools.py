from langchain_core.tools import Tool
from pydantic import BaseModel
from .fetching_description_from_huggingface import fetching_description_from_huggingface

class WebScraperParameters(BaseModel):
    url: str

def web_scraper(url: str) -> str:
    """Get more details about the model using the model url"""
    # scraper = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    # scrape_status = scraper.scrape_url(
    #     url,
    #     formats=['markdown']
    # )
    # return scrape_status.markdown
    result = fetching_description_from_huggingface(url)
    return result

tools = [
    Tool(
        name="web_scrapper",
        func=web_scraper,
        description="Use to get more detail about the model",
        args_schema=WebScraperParameters
    )
]