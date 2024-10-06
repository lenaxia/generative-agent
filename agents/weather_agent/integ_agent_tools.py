import pprint
import unittest
from agents.weather_agent.agent import WeatherInput, WeatherTool, CityToCoordinatesTool, ZipCodeToCoordinatesTool

class WeatherToolIntegrationTest(unittest.TestCase):
    def test_weather_tool(self):
        input_data = WeatherInput(lat=47.6062, lon=-122.3321)
        tool = WeatherTool()
        result = tool._run(input_data)
        print(result)

        self.assertIsNotNone(result)
        self.assertIn("The weather is currently", result)

class CityToCoordinatesToolIntegrationTest(unittest.TestCase):
    def test_city_to_coordinates_tool(self):
        tool = CityToCoordinatesTool()
        result = tool._run("Seattle")

        self.assertIsNotNone(result)
        self.assertIn("lat", result)
        self.assertIn("lon", result)
        self.assertAlmostEqual(float(result["lat"]), 47.6038321, places=4)
        self.assertAlmostEqual(float(result["lon"]), -122.330062, places=4)

class ZipCodeToCoordinatesToolIntegrationTest(unittest.TestCase):
    def test_zipcode_to_coordinates_tool(self):
        tool = ZipCodeToCoordinatesTool()
        result = tool._run("98101")

        self.assertIsNotNone(result)
        self.assertIn("lat", result)
        self.assertIn("lon", result)
        self.assertAlmostEqual(result["lat"], 47.610771136215234, places=4)
        self.assertAlmostEqual(result["lon"], -122.3361975901451, places=4)

if __name__ == '__main__':
    unittest.main()
