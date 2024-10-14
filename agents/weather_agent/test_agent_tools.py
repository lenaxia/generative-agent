import unittest
from unittest.mock import patch, Mock
from agents.weather_agent.agent import WeatherAgent, WeatherInput, WeatherTool, CityToCoordinatesTool, ZipCodeToCoordinatesTool
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus

class TestWeatherAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.llm_factory = LLMFactory({})
        self.message_bus = MessageBus()
        self.agent_id = "weather_agent"
        self.agent = WeatherAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id)

    def test_run(self):
        instruction = "What is the weather forecast for Seattle, WA?"
        expected_output = "The current weather forecast for the location is: Cloudy with a chance of rain."

        # Mock the LLM provider
        mock_provider = Mock()
        mock_provider.invoke.return_value = expected_output
        self.llm_factory.create_provider = Mock(return_value=mock_provider)

        result = self.agent.run(instruction, history=[])
        self.assertEqual(result, expected_output)

    def test_format_input(self):
        instruction = "What is the weather forecast for Boston, MA?"
        formatted_input = self.agent._format_input(instruction)
        self.assertEqual(formatted_input, instruction)

    def test_process_output(self):
        output = "The current weather forecast for the location is: Sunny with a high of 75°F."
        processed_output = self.agent._process_output(output)
        self.assertEqual(processed_output, output)

class TestWeatherTool(unittest.TestCase):
    @patch('agents.weather_agent.agent.requests.get')
    def test_weather_tool(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "properties": {
                "periods": [
                    {
                        "detailedForecast": "Cloudy with a chance of rain."
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        input_data = WeatherInput(lat=47.6062, lon=-122.3321)
        tool = WeatherTool()
        result = tool._run(input_data)

        self.assertEqual(result, "The current weather forecast for the location is: Cloudy with a chance of rain.")

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
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "lat": 40.7128,
                "lon": -74.0059
            }
        ]
        mock_get.return_value = mock_response

        tool = ZipCodeToCoordinatesTool()
        result = tool._run("10001")

        self.assertEqual(result, {"lat": 40.7128, "lon": -74.0059})

if __name__ == '__main__':
    unittest.main()
