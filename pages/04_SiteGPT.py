from langchain_community.document_loaders import SitemapLoader
from fake_useragent import UserAgent
import streamlit as st
import asyncio
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize a UserAgent object
ua = UserAgent()

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    try:
        loader = SitemapLoader(url)
        loader.requests_per_second = 3
        
        # Set a realistic user agent
        loader.headers = {'User-Agent': ua.random}
        docs = loader.load()
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