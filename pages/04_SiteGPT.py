from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fake_useragent import UserAgent
import streamlit as st
import asyncio
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize a UserAgent object
ua = UserAgent()

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/blog\/).*",
            ],
            parsing_function=parse_page
        )
        loader.requests_per_second = 3
        
        # Set a realistic user agent
        loader.headers = {'User-Agent': ua.random}
        docs = loader.load_and_split(text_splitter=splitter)
        logging.debug(f"Loaded documents: {docs}")
        return docs
    except Exception as e:
        logging.error(f"Error loading sitemap: {e}")
        return []

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("Site GPT")

st.markdown(
    """
    # SiteGPT
    
    Ask questions about the content of a website.
    
    Start by writing the URL of the website on the sidebar.
    """
)

if "win32" in sys.platform:
    # Windows specific event-loop policy & cmd
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    cmds = [["C:/windows/system32/HOSTNAME.EXE"]]
else:
    # Unix default event-loop policy & cmds
    cmds = [
        ['du', '-sh', '/Users/fredrik/Desktop'],
        ['du', '-sh', '/Users/fredrik'],
        ['du', '-sh', '/Users/fredrik/Pictures']
    ]

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
        if docs:
            st.write(docs)
        else:
            st.error("Failed to load documents from the sitemap. Please check the URL and try again.")