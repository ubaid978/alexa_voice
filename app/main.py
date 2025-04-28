from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langserve import add_routes
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain import hub
from langchain.tools import Tool
from langchain.utilities import SearchApiAPIWrapper
from langchain.agents import initialize_agent
from youtubesearchpython import VideosSearch
import urllib
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from dotenv import load_dotenv,find_dotenv
import webbrowser
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
import requests
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
import pytz

# Load environment variables
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI(title="AI Assistant Server")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Request model
class AgentRequest(BaseModel):
    input: str

# Tool definitions
def open_notepad(text):
    os.system("start notepad")

def open_youtube(text):
    webbrowser.open("https://www.youtube.com")

def search_youtube(query: str):
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={encoded_query}"
    webbrowser.open(url)

@tool
def say_hello(name: str) -> str:
    """Say hello to someone by name."""
    return f"Hello {name}, I'm Alexa, your assistant."
@tool
def current_time(text):
    'use this toll current time this tool return current time'
    pakistan_timezone = pytz.timezone("Asia/Karachi")
    return f"Current time in Pakistan: {datetime.now(pakistan_timezone).strftime('%H:%M:%S')}"


def serper_search(query: str, ) -> str:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": os.getenv('SERPER_API_KEY'), "Content-Type": "application/json"}
    data = {"q": query}
    response = requests.post(url, headers=headers, json=data)
    
    results = response.json()
    if not results.get("organic"):
        return "No results found."

    output = []
    for i, r in enumerate(results["organic"][:3], 1):
        title = r.get("title", "No title")
        snippet = r.get("snippet", "No snippet")
        link = r.get("link", "")
        output.append(f"{i}. {title}\n{snippet}\nLink: {link}")

    return "\n\n".join(output)





# Initialize LLM
llm = ChatOpenAI(model_name='mistralai/mistral-7b-instruct', temperature=0.7)
os.environ['TAVILY_API_KEY']=os.getenv('TAVILY_API_KEY')
# Initialize tools
react_prompt = hub.pull("hwchase17/react")
search = SearchApiAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
Tavily=TavilySearchResults(api_wrapper=TavilySearchAPIWrapper())

tools = [
   
    Tool(
        name="notepad",
        func=open_notepad,
        description="useful open notepad tool",
    ),
    Tool(
        name="youtube",
        func=open_youtube,
        description="useful open youtube tool",
    ),
    Tool(
        name="youtube_search",
        func=search_youtube,
        description="Searches for on YouTube and opens it in the browser",
    ),
    
    say_hello,
    current_time,
    
    Tool(
        name='Wikipedia',
        func=wikipedia.run,
        description='usefull for quick access to Wikipedia articles or searching for information'
    ),
    Tool(
    name="Google Search via Serper",
    func=lambda q: serper_search(q),
    description="Useful for real-time Google searches for current information and general knowledge and current time information this tool input as query and output search results"
),
    Tool(
        name='TAVILY Search',
        func=Tavily.run,
        description='useful for searching TAVILY then useful for real time search results like current weather like current weather conditions and current news'

    )

]

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Serve HTML template
@app.get("/")
async def serve_ui(request: Request):
    return templates.TemplateResponse("voice.html", {"request": request})

# Agent endpoint
@app.post("/agent")
async def agent_response(request: AgentRequest):
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        verbose=True,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=react_prompt,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": "Your name is Alexa. You are a smart, helpful, and friendly AI assistant.",
        }
    )
    
    try:
        response = agent_executor.run(request.input)
        return {'user': request.input, 'agent': response}
    except Exception as e:
        return {'user': request.input, 'agent': f"Sorry, I encountered an error: {str(e)}"}

if __name__=="__main__":
    uvicorn.run(app,port=1535)

