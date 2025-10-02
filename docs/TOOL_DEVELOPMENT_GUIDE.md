
# Tool Development Guide

## Overview

This guide explains how to create new @tool functions for the StrandsAgent Universal Agent system. Tools are the primary way to extend system capabilities and integrate with external services.

## Tool Architecture

### @tool Decorator Pattern

The system uses the StrandsAgent `@tool` decorator to create reusable functions that can be called by the Universal Agent:

```python
from strands import tool
from typing import Dict, List, Optional

@tool
def example_tool(input_param: str, optional_param: int = 10) -> Dict:
    """
    Example tool that demonstrates the basic pattern.
    
    Args:
        input_param: Required string parameter
        optional_param: Optional integer parameter with default value
        
    Returns:
        Dict containing the tool execution result
    """
    # Tool implementation
    result = {
        "status": "success",
        "data": f"Processed: {input_param}",
        "metadata": {
            "param_value": optional_param,
            "timestamp": time.time()
        }
    }
    return result
```

### Tool Categories

Tools are organized into logical categories based on functionality:

- **Planning Tools** (`llm_provider/planning_tools.py`): Task planning and optimization
- **Search Tools** (`llm_provider/search_tools.py`): Information retrieval and search
- **Weather Tools** (`llm_provider/weather_tools.py`): Weather data and forecasting
- **Summarizer Tools** (`llm_provider/summarizer_tools.py`): Text processing and summarization
- **Slack Tools** (`llm_provider/slack_tools.py`): Team communication and collaboration

## Creating New Tools

### Step 1: Define Tool Function

Create a new tool function with proper type hints and documentation:

```python
# llm_provider/custom_tools.py
from strands import tool
from typing import Dict, List, Optional
import requests
import logging

logger = logging.getLogger(__name__)

@tool
def analyze_sentiment(text: str, language: str = "en") -> Dict:
    """
    Analyze the sentiment of the provided text.
    
    Args:
        text: Text to analyze for sentiment
        language: Language code for analysis (default: "en")
        
    Returns:
        Dict containing sentiment analysis results
        
    Raises:
        ValueError: If text is empty or language is unsupported
        ConnectionError: If sentiment analysis service is unavailable
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    supported_languages = ["en", "es", "fr", "de", "it"]
    if language not in supported_languages:
        raise ValueError(f"Unsupported language: {language}")
    
    try:
        # Example external API call
        response = requests.post(
            "https://api.sentiment-service.com/analyze",
            json={
                "text": text,
                "language": language
            },
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        result = {
            "status": "success",
            "sentiment": data["sentiment"],  # positive, negative, neutral
            "confidence": data["confidence"],  # 0.0 to 1.0
            "language": language,
            "metadata": {
                "text_length": len(text),
                "processing_time": data.get("processing_time", 0),
                "timestamp": time.time()
            }
        }
        
        logger.info(f"Sentiment analysis completed: {result['sentiment']} ({result['confidence']:.2f})")
        return result
        
    except requests.RequestException as e:
        logger.error(f"Sentiment analysis API error: {e}")
        raise ConnectionError(f"Failed to connect to sentiment analysis service: {e}")
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise
```

### Step 2: Register Tool

Register the tool with the ToolRegistry:

```python
# llm_provider/tool_registry.py
from .custom_tools import analyze_sentiment

class ToolRegistry:
    def __init__(self):
        self.tools = {
            # Existing tools
            "create_task_plan": create_task_plan,
            "web_search": web_search,
            "get_weather": get_weather,
            "summarize_text": summarize_text,
            "send_slack_message": send_slack_message,
            
            # New custom tool
            "analyze_sentiment": analyze_sentiment,
        }
```

### Step 3: Add Tool to Role Configuration

Configure which roles can use the new tool:

```yaml
# config.yaml
universal_agent:
  role_tools:
    # Add to existing roles
    summarizer:
      - summarize_text
      - extract_key_points
      - analyze_sentiment  # New tool added to summarizer role
    
    # Or create new role
    analyst:
      - analyze_sentiment
      - custom_analysis_tool
```

### Step 4: Write Tests

Create comprehensive tests for the new tool:

```python
# tests/llm_provider/test_custom_tools.py
import unittest
from unittest.mock import patch, Mock
import pytest
from llm_provider.custom_tools import analyze_sentiment

class TestCustomTools(unittest.TestCase):
    
    def test_analyze_sentiment_success(self):
        """Test successful sentiment analysis"""
        with patch('requests.post') as mock_post:
            # Mock successful API response
            mock_response = Mock()
            mock_response.json.return_value = {
                "sentiment": "positive",
                "confidence": 0.85,
                "processing_time": 0.5
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            result = analyze_sentiment("I love this product!")
            
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["sentiment"], "positive")
            self.assertEqual(result["confidence"], 0.85)
            self.assertEqual(result["language"], "en")
    
    def test_analyze_sentiment_empty_text(self):
        """Test error handling for empty text"""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyze_sentiment("")
    
    def test_analyze_sentiment_unsupported_language(self):
        """Test error handling for unsupported language"""
        with pytest.raises(ValueError, match="Unsupported language"):
            analyze_sentiment("Hello", language="xyz")
    
    def test_analyze_sentiment_api_error(self):
        """Test error handling for API failures"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.RequestException("API Error")
            
            with pytest.raises(ConnectionError, match="Failed to connect"):
                analyze_sentiment("Test text")
    
    def test_analyze_sentiment_with_different_languages(self):
        """Test sentiment analysis with different languages"""
        test_cases = [
            ("Hello world", "en"),
            ("Hola mundo", "es"),
            ("Bonjour monde", "fr")
        ]
        
        for text, lang in test_cases:
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "sentiment": "neutral",
                    "confidence": 0.7
                }
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response
                
                result = analyze_sentiment(text, language=lang)
                self.assertEqual(result["language"], lang)

if __name__ == "__main__":
    unittest.main()
```

## Tool Development Best Practices

### 1. Function Design

**Clear Function Signatures:**
```python
@tool
def good_tool(required_param: str, optional_param: int = 10) -> Dict:
    """Clear, descriptive function with proper types"""
    pass

# Avoid unclear signatures
@tool  
def bad_tool(param):  # No type hints, unclear parameters
    pass
```

**Comprehensive Documentation:**
```python
@tool
def comprehensive_tool(data: str, options: Dict = None) -> Dict:
    """
    One-line summary of what the tool does.
    
    Detailed description of the tool's functionality, use cases,
    and any important behavior notes.
    
    Args:
        data: Description of the required parameter
        options: Description of optional parameter with examples
            - key1: Description of option
            - key2: Description of another option
            
    Returns:
        Dict containing:
            - status: Operation status ("success" or "error")
            - data: Main result data
            - metadata: Additional information about the operation
            
    Raises:
        ValueError: When input validation fails
        ConnectionError: When external service is unavailable
        TimeoutError: When operation exceeds time limit
        
    Examples:
        >>> result = comprehensive_tool("sample data")
        >>> print(result["status"])
        "success"
        
        >>> result = comprehensive_tool("data", {"format": "json"})
        >>> print(result["data"])
        {...}
    """
    pass
```

### 2. Error Handling

**Robust Error Handling:**
```python
@tool
def robust_tool(input_data: str) -> Dict:
    """Tool with comprehensive error handling"""
    try:
        # Input validation
        if not input_data or not input_data.strip():
            raise ValueError("Input data cannot be empty")
        
        # External service call with timeout
        response = requests.get(
            f"https://api.service.com/process",
            params={"data": input_data},
            timeout=30
        )
        response.raise_for_status()
        
        # Process response
        data = response.json()
        
        return {
            "status": "success",
            "data": data,
            "metadata": {
                "processing_time": response.elapsed.total_seconds(),
                "timestamp": time.time()
            }
        }
        
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        return {
            "status": "error",
            "error_type": "validation_error",
            "message": str(e)
        }
    
    except requests.RequestException as e:
        logger.error(f"External service error: {e}")
        return {
            "status": "error",
            "error_type": "service_error",
            "message": f"External service unavailable: {e}"
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in robust_tool: {e}")
        return {
            "status": "error",
            "error_type": "internal_error",
            "message": "Internal tool error occurred"
        }
```

### 3. Configuration Integration

**Tool Configuration:**
```python
# llm_provider/configurable_tools.py
from config.config_manager import get_global_config

@tool
def configurable_tool(query: str) -> Dict:
    """Tool that uses system configuration"""
    config = get_global_config()
    
    # Get tool-specific configuration
    tool_config = config.get('tools', {}).get('configurable_tool', {})
    timeout = tool_config.get('timeout', 30)
    max_results = tool_config.get('max_results', 10)
    
    # Use configuration in tool logic
    try:
        response = requests.get(
            "https://api.service.com/search",
            params={
                "q": query,
                "limit": max_results
            },
            timeout=timeout
        )
        # Process response...
        
    except requests.Timeout:
        return {
            "status": "error",
            "error_type": "timeout",
            "message": f"Request timed out after {timeout}s"
        }
```

**Configuration Schema:**
```yaml
# config.yaml
tools:
  configurable_tool:
    timeout: 45
    max_results: 20
    api_endpoint: "https://api.service.com"
    
  analyze_sentiment:
    service_url: "https://sentiment-api.com"
    timeout: 30
    cache_results: true
    cache_ttl: 3600
```

## Advanced Tool Patterns

### 1. Stateful Tools

Tools that maintain state across calls:

```python
class StatefulToolManager:
    def __init__(self):
        self.session_data = {}
    
    @tool
    def start_session(self, session_id: str, config: Dict) -> Dict:
        """Start a new session with configuration"""
        self.session_data[session_id] = {
            "config": config,
            "created_at": time.time(),
            "calls": 0
        }
        return {"status": "success", "session_id": session_id}
    
    @tool
    def process_with_session(self, session_id: str, data: str) -> Dict:
        """Process data using session context"""
        if session_id not in self.session_data:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.session_data[session_id]
        session["calls"] += 1
        
        # Use session configuration
        config = session["config"]
        # Process data with session context...
        
        return {
            "status": "success",
            "data": processed_data,
            "session_info": {
                "calls": session["calls"],
                "session_age": time.time() - session["created_at"]
            }
        }

# Global instance
stateful_manager = StatefulToolManager()
start_session = stateful_manager.start_session
process_with_session = stateful_manager.process_with_session
```

### 2. Async Tools

Tools that handle asynchronous operations:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

@tool
def async_tool(urls: List[str]) -> Dict:
    """Tool that processes multiple URLs concurrently"""
    
    async def fetch_url(session, url):
        try:
            async with session.get(url, timeout=10) as response:
                return {
                    "url": url,
                    "status": response.status,
                    "content": await response.text()
                }
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "error": str(e)
            }
    
    async def process_urls():
        import aiohttp
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
    
    # Run async operation
    try:
        results = asyncio.run(process_urls())
        return {
            "status": "success",
            "results": results,
            "total_processed": len(urls)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
```

### 3. Streaming Tools

Tools that provide streaming responses:

```python
@tool
def streaming_tool(query: str) -> Dict:
    """Tool that provides streaming responses"""
    
    def generate_stream():
        # Simulate streaming data
        for i in range(10):
            yield {
                "chunk": i,
                "data": f"Processing chunk {i} for query: {query}",
                "timestamp": time.time()
            }
            time.sleep(0.1)  # Simulate processing time
    
    # Collect streaming results
    chunks = list(generate_stream())
    
    return {
        "status": "success",
        "streaming_data": chunks,
        "total_chunks": len(chunks),
        "query": query
    }
```

## Tool Integration

### 1. Role-Based Tool Assignment

Configure tools for specific Universal Agent roles:

```python
# llm_provider/tool_registry.py
class ToolRegistry:
    def __init__(self):
        self.role_tools = {
            "planning": [
                "create_task_plan",
                "validate_dependencies",
                "optimize_execution_order"
            ],
            "search": [
                "web_search",
                "search_with_filters",
                "analyze_search_results"
            ],
            "analyst": [  # New role
                "analyze_sentiment",
                "extract_entities",
                "classify_content"
            ]
        }
    
    def get_tools_for_role(self, role: str) -> List[str]:
        """Get available tools for a specific role"""
        return self.role_tools.get(role, [])
    
    def register_tool_for_role(self, role: str, tool_name: str):
        """Register a tool for a specific role"""
        if role not in self.role_tools:
            self.role_tools[role] = []
        
        if tool_name not in self.role_tools[role]:
            self.role_tools[role].append(tool_name)
```

### 2. Dynamic Tool Loading

Load tools dynamically based on configuration:

```python
# llm_provider/dynamic_tool_loader.py
import importlib
from typing import Dict, Any

class DynamicToolLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loaded_tools = {}
    
    def load_tool_module(self, module_path: str):
        """Load tools from a Python module"""
        try:
            module = importlib.import_module(module_path)
            
            # Find all @tool decorated functions
            tools = {}
            for name in dir(module):
                obj = getattr(module, name)
                if hasattr(obj, '_tool_metadata'):  # StrandsAgent tool marker
                    tools[name] = obj
            
            self.loaded_tools.update(tools)
            return tools
            
        except ImportError as e:
            logger.error(f"Failed to load tool module {module_path}: {e}")
            return {}
    
    def load_configured_tools(self):
        """Load all tools specified in configuration"""
        tool_modules = self.config.get('tool_modules', [])
        
        for module_path in tool_modules:
            self.load_tool_module(module_path)
        
        return self.loaded_tools
```

### 3. Tool Middleware

Add middleware for cross-cutting concerns:

```python
# llm_provider/tool_middleware.py
import time
import logging
from functools import wraps

def tool_middleware(func):
    """Middleware decorator for tools"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        tool_name = func.__name__
        
        logger.info(f"Tool {tool_name} started with args: {args[:2]}...")  # Log first 2 args
        
        try:
            # Execute tool
            result = func(*args, **kwargs)
            
            # Add middleware metadata
            if isinstance(result, dict):
                result.setdefault("metadata", {})
                result["metadata"]["execution_time"] = time.time() - start_time
                result["metadata"]["tool_name"] = tool_name
            
            logger.info(f"Tool {tool_name} completed successfully in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed after {time.time() - start_time:.2f}s: {e}")
            
            # Return standardized error response
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e),
                "metadata": {
                    "tool_name": tool_name,
                    "execution_time": time.time() - start_time,
                    "timestamp": time.time()
                }
            }
    
    return wrapper

# Apply middleware to tools
@tool_middleware
@tool
def monitored_tool(param: str) -> Dict:
    """Tool with automatic monitoring"""
    # Tool implementation
    return {"status": "success", "data": param}
```

## Testing Tools

### Unit Testing

```python
# tests/llm_provider/test_tool_template.py
import unittest
from unittest.mock import patch, Mock
import pytest
from llm_provider.custom_tools import your_new_tool

class TestYourNewTool(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.valid_input = "test input"
        self.invalid_input = ""
    
    def test_tool_success_case(self):
        """Test successful tool execution"""
        result = your_new_tool(self.valid_input)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("data", result)
        self.assertIn("metadata", result)
    
    def test_tool_input_validation(self):
        """Test input validation"""
        with pytest.raises(ValueError):
            your_new_tool(self.invalid_input)
    
    @patch('requests.get')
    def test_tool_external_service_mock(self, mock_get):
        """Test tool with mocked external service"""
        # Configure mock
        mock_response = Mock()
        mock_response.json.return_value = {"result": "mocked"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = your_new_tool(self.valid_input)
        
        # Verify mock was called correctly
        mock_get.assert_called_once()
        self.assertEqual(result["status"], "success")
    
    def test_tool_error_handling(self):
        """Test error handling scenarios"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")
            
            result = your_new_tool(self.valid_input)
            
            self.assertEqual(result["status"], "error")
            self.assertIn("error_type", result)
    
    def test_tool_performance(self):
        """Test tool performance characteristics"""
        start_time = time.time()
        result = your_new_tool(self.valid_input)
        execution_time = time.time() - start_time
        
        # Assert reasonable execution time
        self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds
        
        # Check if metadata includes timing
        if "metadata" in result:
            self.assertIn("execution_time", result["metadata"])
```

### Integration Testing

```python
# tests/integration/test_tool_integration.py
import unittest
from supervisor.workflow_engine import WorkflowEngine
from llm_provider.factory import LLMFactory
from common.message_bus import MessageBus

class TestToolIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test environment"""
        self.message_bus = MessageBus()
        self.llm_factory = LLMFactory(test_configs)
        self.workflow_engine = WorkflowEngine(self.llm_factory, self.message_bus)
    
    def test_tool_in_workflow(self):
        """Test tool execution within a workflow"""
        instruction = "Use the new custom tool to analyze sentiment of 'I love this system'"
        
        workflow_id = self.workflow_engine.start_workflow(instruction)
        
        # Wait for completion
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.workflow_engine.get_workflow_status(workflow_id)
            if status["state"] in ["COMPLETED", "FAILED"]:
                break
            time.sleep(0.5)
        
        # Verify workflow completed successfully
        final_status = self.workflow_engine.get_workflow_status(workflow_id)
        self.assertEqual(final_status["state"], "COMPLETED")
        
        # Verify tool was used
        results = final_status.get("results", {})
        self.assertIn("sentiment_analysis", results)
    
    def test_tool_with_universal_agent(self):
        """Test tool usage with Universal Agent role assumption"""
        from llm_provider.universal_agent import UniversalAgent
        
        universal_agent = UniversalAgent(self.llm_factory)
        
        # Assume role that has access to the tool
        agent = universal_agent.assume_role(
            role="analyst",
            tools=["analyze_sentiment"]
        )
        
        # Test tool execution through agent
        result = agent("Analyze the sentiment of: 'This is amazing!'")
        
        # Verify result contains sentiment analysis
        self.assertIn("sentiment", result.lower())
```

## Tool Documentation

### 1. Inline Documentation

Use comprehensive docstrings with examples:

```python
@tool
def well_documented_tool(text: str, format: str = "json") -> Dict:
    """
    Process text and return results in specified format.
    
    This tool processes input text using advanced NLP techniques and returns
    structured results. It supports multiple output formats and provides
    detailed metadata about the processing.
    
    Args:
        text (str): Input text to process. Must be non-empty and contain
                   at least 10 characters for meaningful analysis.
        format (str, optional): Output format. Supported values:
                               - "json": Structured JSON output (default)
                               - "xml": XML formatted output
                               - "plain": Plain text summary
    
    Returns:
        Dict: Processing results containing:
            - status (str): "success" or "error"
            - data (Any): Processed results in requested format
            - metadata (Dict): Processing metadata including:
                - processing_time (float): Execution time in seconds
                - input_length (int): Length of input text
                - output_format (str): Requested output format
                - confidence (float): Processing confidence score
    
    Raises:
        ValueError: If text is empty or format is unsupported
        ProcessingError: If text processing fails
        TimeoutError: If processing exceeds time limit
    
    Examples:
        Basic usage:
        >>> result = well_documented_tool("Hello world")
        >>> print(result["status"])
        "success"
        
        With custom format:
        >>> result = well_documented_tool("Hello world", format="xml")
        >>> print(result["data"])
        "<result>...</result>"
        
        Error handling:
        >>> result = well_documented_tool("")
        ValueError: Text cannot be empty
    
    Note:
        This tool requires an active internet connection for advanced
        processing features. In offline mode, basic processing is available.
    """
    # Implementation...
```

### 2. External Documentation

Create separate documentation files for complex tools:

```markdown
# docs/tools/sentiment_analysis_tool.md

# Sentiment Analysis Tool

## Overview
The sentiment analysis tool provides advanced text sentiment analysis with support for multiple languages and confidence scoring.

## Usage

### Basic Usage
```python
result = analyze_sentiment("I love this product!")
```

### Advanced Usage
```python
result = analyze_sentiment(
    text="Me encanta este producto",
    language="es"
)
```

## Configuration

Add to config.yaml:
```yaml
tools:
  analyze_sentiment:
    service_url: "https://sentiment-api.com"
    timeout: 30
    supported_languages: ["en", "es", "fr", "de"]
```

## API Reference

### Parameters
- `text` (str, required): Text to analyze
- `language` (str, optional): Language code (default: "en")

### Returns
- `sentiment` (str): "positive", "negative", or "neutral"
- `confidence` (float): Confidence score 0.0-1.0
- `language` (str): Detected or specified language

## Error Handling
- `ValueError`: Invalid input parameters
- `ConnectionError`: Service unavailable
- `TimeoutError`: Request timeout

## Examples

See `examples/sentiment_analysis_examples.py` for complete usage examples.
```

## Tool Deployment

### 1. Tool Registration

Register new tools in the system:

```python
# llm_provider/tool_registry.py
from .custom_tools import analyze_sentiment, extract_entities
from .advanced_tools import complex_analysis_tool

class ToolRegistry:
    def __init__(self):
        self.tools = {
            # Core tools
            "create_task_plan": create_task_plan,
            "web_search": web_search,
            
            # Custom tools
            "analyze_sentiment": analyze_sentiment,
            "extract_entities": extract_entities,
            "complex_analysis": complex_analysis_tool,
        }
        
        # Role assignments
        self.role_tools = {
            "analyst": [
                "analyze_sentiment",
                "extract_entities",
                "complex_analysis"
            ]
        }
    
    def register_tool(self, name: str, tool_func, roles: List[str] = None):
        """Register a new tool"""
        self.tools[name] = tool_func
        
        if roles:
            for role in roles:
                if role not in self.role_tools:
                    self.role_tools[role] = []
                self.role_tools[role].append(name)
```

### 2. Configuration Updates

Update system configuration to include new tools:

```yaml
# config.yaml
universal_agent:
  role_tools:
    analyst:  # New role for custom tools
      - analyze_sentiment
      - extract_entities
      - complex_analysis
    
    # Add tools to existing roles
    summarizer:
      - summarize_text
      - extract_key_points
      - analyze_sentiment  # Add sentiment analysis to summarizer

# Tool-specific configuration
tools:
  analyze_sentiment:
    service_url: ${SENTIMENT_API_URL:https://api.sentiment.com}
    timeout: 30
    cache_enabled: true
    
  extract_entities:
    service_url: ${NER_API_URL:https://api.entities.com}
    timeout: 45
    entity_types: ["PERSON", "ORG", "LOCATION"]
```

### 3. Testing Deployment

Test new tools in the system:

```python
# scripts/test_new_tools.py
from supervisor.workflow_engine import WorkflowEngine
from llm_provider.factory import LLMFactory
from common.message_bus import MessageBus

def test_new_tool_deployment():
    """Test deployment of new tools"""
    
    # Initialize system
    message_bus = MessageBus()
    llm_factory = LLMFactory(configs)
    workflow_engine = WorkflowEngine(llm_factory, message_bus)
    
    # Test tool availability
    from llm_provider.tool_registry import ToolRegistry
    registry = ToolRegistry()
    
    # Check if new tools are registered
    available_tools = registry.get_available_tools()
    new_tools = ["analyze_sentiment", "extract_entities"]
    
    for tool in new_tools:
        if tool in available_tools:
            print(f"✅ Tool {tool} is registered")
        else:
            print(f"❌ Tool {tool} is not registered")
    
    # Test tool execution in workflow
    test_instruction = "Analyze the sentiment of this text: 'I am very happy today!'"
    
    try:
        workflow_id = workflow_engine.start_workflow(test_instruction)
        print(f"✅ Workflow with new tool started: {workflow_id}")
        
        # Monitor workflow completion
        # ... monitoring code ...
        
    except Exception as e:
        print(f"❌ Tool deployment test failed: {e}")

if __name__ == "__main__":
    test_new_tool_deployment()
```

## Tool Performance Optimization

### 1. Caching

Implement caching for expensive operations:

```python
from functools import lru_cache
import hashlib

@tool
def cached_tool(input_data: str, cache_ttl: int = 3600) -> Dict:
    """Tool with built-in caching"""
    
    # Create cache key
    cache_key = hashlib.md5(input_data.encode()).hexdigest()
    
    # Check cache
    cached_result = get_from_cache(cache_key)
    if cached_result and not cache_expired(cached_result, cache_ttl):
        return {
            "status": "success",
            "data": cached_result["data"],
            "metadata": {
                