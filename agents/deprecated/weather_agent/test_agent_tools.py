import unittest
from typing import Dict
from unittest.mock import patch, Mock
from agents.base_agent import AgentInput
from agents.weather_agent.agent import WeatherAgent, WeatherTool, CityToCoordinatesTool, ZipCodeToCoordinatesTool
from llm_provider.factory import LLMFactory
from common.message_bus import MessageBus

class TestWeatherAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.llm_factory = LLMFactory({})
        self.message_bus = MessageBus()
        self.agent_id = "weather_agent"
        self.agent = WeatherAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id)

    def test_format_input(self):
        prompt = "What is the weather forecast for Boston, MA?"
        history = ["This is some maybe relevant history", "This is some more history"]
        formatted_input = self.agent._format_input({"prompt": prompt, "history": history})
        expected_output = AgentInput(prompt=prompt, history=history)
        self.assertEqual(formatted_input, expected_output)

    def test_process_output(self):
        input: Dict = dict(
            task_id="task_1",
            task_name="fetch_data",
            request_id="request_1",
            agent_id="agent_1",
            task_type="fetch_data",
            prompt="Fetch data from API",
            status="pending",
            inbound_edges=[],
            outbound_edges=[],
            include_full_history=False,
        )
        output = "The current weather forecast for the location is: Sunny with a high of 75°F."
        processed_output = self.agent._process_output(input, output)
        self.assertEqual(processed_output, output)

class TestWeatherTool(unittest.TestCase):
    @patch('agents.weather_agent.agent.requests.get')
    def test_weather_tool(self, mock_get):
        # Define the responses for each mocked GET call
        points_response = {
            "properties": {
                "forecast": "https://api.weather.gov/gridpoints/SEW/125,68/forecast"
            }
        }
        forecast_response = {
            "properties": {
                "periods": [
                    {
                        "detailedForecast": "Sunny with a high of 75F"
                    }
                ]   
            }
        }

        mock_get.side_effect = [
            Mock(json=Mock(return_value=points_response)),
            Mock(json=Mock(return_value=forecast_response))
        ]

        input_data = {"lat": 47.6062, "lon": -122.3321}
        tool = WeatherTool()
        result = tool._run(input_data)

        self.assertEqual(result, "The weather is currently Sunny with a high of 75F")

class TestCityToCoordinatesTool(unittest.TestCase):
    @patch('agents.weather_agent.agent.requests.get')
    def test_city_to_coordinates_tool(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "lat": 47.6062,
                "lon": -122.3321
            }
        ]
        mock_get.return_value = mock_response

        tool = CityToCoordinatesTool()
        result = tool._run("Seattle, WA")

        self.assertEqual(result, {"lat": 47.6062, "lon": -122.3321})

class TestZipCodeToCoordinatesTool(unittest.TestCase):
    @patch('agents.weather_agent.agent.requests.get')
    def test_zipcode_to_coordinates_tool(self, mock_get):  
        zipcode_response = [
            {
                "place_id": 331090693,
                "licence": "Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright",
                "lat": "47.60164925272353",
                "lon": "-122.32918928345323",
                "class": "place",
                "type": "postcode",
                "place_rank": 21,
                "importance": 0.12000999999999995,
                "addresstype": "postcode",
                "name": "98104",
                "display_name": "98104, Seattle, King County, Washington, United States",
                "boundingbox": [
                    "47.5516493",
                    "47.6516493",
                    "-122.3791893",
                    "-122.2791893"
                ]
            }
        ]
        forecast_response = {
            "properties": {
                "periods": [
                    {
                        "detailedForecast": "Sunny with a high of 75F"
                    }
                ]   
            }
        }

        mock_get.side_effect = [
            Mock(json=Mock(return_value=zipcode_response)),
            Mock(json=Mock(return_value=forecast_response))
        ]

        tool = ZipCodeToCoordinatesTool()
        result = tool._run("10001", "us")

        self.assertEqual(result, {"lat": 47.60164925272353, "lon": -122.32918928345323})

if __name__ == '__main__':
    unittest.main()
