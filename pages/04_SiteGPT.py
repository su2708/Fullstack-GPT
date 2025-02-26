from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from fake_useragent import UserAgent
import streamlit as st
import asyncio
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize a UserAgent object
ua = UserAgent()

answers_prompt = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question. If you can't just say you don't know, Don't make anything up.
    
    Then, give a score to the answer between 0 and 5.
    
    If the answer answers the user question the score should be high, else it should be low.
    
    Make sure to always include the answer's score even if it's 0.
    
    Context: {context}
    
    Examples:
    
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
    
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
    
    Your turn!
    
    Question: {question}
""")

llm = ChatOpenAI(
    temperature=0.1,
)

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    answers = []
    for doc in docs:
        result = answers_chain.invoke({
            "context": doc.page_content,
            "question": question
        })
        answers.append(result.content)
    st.write(answers)

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

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        loader = SitemapLoader(
            url,
            parsing_function=parse_page
        )
        loader.requests_per_second = 3
        
        # Set a realistic user agent
        loader.headers = {'User-Agent': ua.random}
        docs = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        logging.debug(f"Loaded documents: {docs}")
        return vector_store.as_retriever()
    except Exception as e:
        logging.error(f"Error loading sitemap: {e}")
        return []

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

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
        retriever = load_website(url)
        if retriever:
            docs = retriever.invoke("What is the price of GPT-4?")
            
            chain = {
                "docs": retriever, 
                "question": RunnablePassthrough()
            } | RunnableLambda(get_answers)
            
            chain.invoke("Who will use zep?")
        else:
            st.error("Failed to load documents from the sitemap. Please check the URL and try again.")