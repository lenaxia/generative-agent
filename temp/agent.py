# Import relevant functionality
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain.tools import BaseTool
from typing import Dict, Any
from datetime import datetime
import requests

class WeatherTool(BaseTool):
    name: str = "check_weather"
    description: str = "Return the current weather for the specified coordinates."

    def _run(self, input_dict: Dict[str, Any]) -> str:
        if "lat" not in input_dict or "lon" not in input_dict:
            return "Invalid input. Please provide valid latitude and longitude coordinates."

        lat = input_dict["lat"]
        lon = input_dict["lon"]

        # Fetch weather data from NOAA API
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {
            'User-Agent': 'TheKaoCloud/1.0',
            'Referer': 'https://thekao.cloud'
        }

        points_raw = requests.get(points_url,headers=headers)
        points_response = points_raw.json()

        if "properties" not in points_response:
            return "Sorry, I couldn't fetch the weather data for that location."

        forecast_url = points_response["properties"]["forecast"]
        forecast_raw = requests.get(forecast_url)
        forecast_response = forecast_raw.json()

        if "properties" not in forecast_response or "periods" not in forecast_response["properties"]:
            return "Sorry, I couldn't fetch the weather forecast for that location."

        current_weather = forecast_response["properties"]["periods"][0]["detailedForecast"]
        return f"The weather is currently {current_weather}"


class CityToCoordinatesTool(BaseTool):
    name: str = "city_to_coordinates"
    description: str = "Convert a city name to latitude and longitude coordinates. Call this in order to convert a city name into coordinates for fetching the weather"

    def _run(self, city: str) -> Dict[str, float]:
        url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json"

        # Set custom headers
        headers = {
            'User-Agent': 'TheKaoCloud/1.0',
            'Referer': 'https://thekao.cloud'
        }

        response = requests.get(url, headers=headers)
        response_json = response.json()

        if not response_json:
            raise ValueError(f"Sorry, I couldn't find coordinates for '{city}'.")

        lat = response_json[0]["lat"]
        lon = response_json[0]["lon"]
        return {"lat": lat, "lon": lon}

    def _arun(self, city: str) -> Dict[str, float]:
        raise NotImplementedError("Asynchronous execution not supported.")

# Create the agent
memory = MemorySaver()
#model = ChatOpenAI(
#    base_url="http://192.168.5.74:8080/v1/",
#    model_name="qwen2.5-32b-instruct",
#)
model = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs=dict(temperature=0)
    )
search = TavilySearchResults(max_results=2)
tools = [search, WeatherTool(), CityToCoordinatesTool()]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
#for chunk in agent_executor.stream(
#    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
#):
#    print(chunk)
#    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in Seattle?")]}, config
):
    print(chunk)
    print("----")

