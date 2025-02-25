from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
import streamlit as st
import asyncio
import sys

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

html2text_transformer = Html2TextTransformer()

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
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)