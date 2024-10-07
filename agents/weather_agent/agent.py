import requests
from datetime import datetime
from typing import Any, Dict
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from agents.base_agent import BaseAgent
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    lat: float = Field(..., description="Latitude of the location")
    lon: float = Field(..., description="Longitude of the location")

class WeatherTool(BaseTool):
    name: str = "check_weather"
    description: str = "Return the current weather for the specified coordinates."

    def _run(self, input_dict: Dict[str, Any], at_time: datetime | None = None) -> str:
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

    def _arun(self, input: WeatherInput, at_time: datetime | None = None) -> str:
        raise NotImplementedError("Asynchronous execution not supported.")

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

class ZipCodeToCoordinatesTool(BaseTool):
    name: str = "zipcode_to_coordinates"
    description: str = "Convert a ZIP code to latitude and longitude coordinates for the United States. Call this in order to convert a US zip code into coordinates for fetching the weather"

    def _run(self, zipcode: str) -> Dict[str, float]:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={zipcode}&format=json"

        # Set custom headers
        headers = {
            'User-Agent': 'TheKaoCloud/1.0',
            'Referer': 'https://thekao.cloud'
        }

        response = requests.get(url, headers=headers)
        response_json = response.json()

        if not response_json:
            raise ValueError(f"Sorry, I couldn't find coordinates for ZIP code '{zipcode}' in the United States.")

        # Filter results to get the coordinates for the ZIP code in the United States
        us_result = next((result for result in response_json if "United States" in result["display_name"]), None)

        if not us_result:
            raise ValueError(f"Sorry, I couldn't find coordinates for ZIP code '{zipcode}' in the United States.")

        lat = us_result["lat"]
        lon = us_result["lon"]
        return {"lat": float(lat), "lon": float(lon)}

    def _arun(self, zipcode: str) -> Dict[str, float]:
        raise NotImplementedError("Asynchronous execution not supported.")

class WeatherAgent(BaseAgent):
    def __init__(self, logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)

    @property
    def tools(self):
        return [
            WeatherTool(),
            CityToCoordinatesTool(),
            ZipCodeToCoordinatesTool()
        ]

    def _select_llm_provider(self, llm_type: LLMType, **kwargs):
        llm = self.llm_factory.create_chat_model(llm_type)
        return llm

    def _run(self, llm_provider, instruction: str) -> Any:
        system_prompt = "You are a weather bot who can look up the current weather in a city or zip code. Answer the user query below"

        memory = MemorySaver()
        config = {"configurable": {"thread_id": "abc123"}}
        llm = self.llm_factory.create_chat_model(LLMType.DEFAULT)
        graph = create_react_agent(llm, tools=self.tools)
        inputs = {"messages": [("system", system_prompt),("user", instruction)]}
        output = None
        for chunk in graph.stream(inputs, config):
            output = chunk
            print(chunk)
            print("-----")

        #message = graph.invoke(inputs)
        return output

    def _arun(self, llm_provider, instruction: str) -> Any:
        raise NotImplementedError("Asynchronous execution not supported.")

    def _format_input(self, instruction: str, *args, **kwargs) -> str:
        return instruction

    def _process_output(self, output: str, *args, **kwargs) -> str:
        return output

    def setup(self):
        pass

    def teardown(self):
        pass
