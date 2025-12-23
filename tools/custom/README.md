# Custom Tools

**Add your custom tools here!**

This directory is for user-defined tools that extend the agent's capabilities.

## Quick Start

1. Create a new Python file: `tools/custom/my_tool.py`
2. Define your tool function with the `@tool` decorator
3. That's it! The system will automatically discover and load it.

See `example.py` for a template.

## Example

```python
"""My Custom Tool"""

from strands import tool

@tool
def calculate_roi(investment: float, return_value: float) -> dict:
    """Calculate return on investment.

    Args:
        investment: Initial investment amount
        return_value: Final return value

    Returns:
        Dict with ROI percentage and profit
    """
    profit = return_value - investment
    roi_percentage = (profit / investment) * 100

    return {
        "success": True,
        "roi_percentage": roi_percentage,
        "profit": profit,
        "investment": investment,
        "return": return_value
    }
```

## Best Practices

### 1. Use Type Hints
```python
def my_tool(input: str, count: int = 5) -> dict:
    pass
```

### 2. Return Structured Data
Always return a dict with at minimum a `success` field:
```python
return {
    "success": True,
    "result": "...",
    "metadata": {...}
}
```

### 3. Handle Errors
```python
try:
    result = do_something()
    return {"success": True, "result": result}
except Exception as e:
    return {"success": False, "error": str(e)}
```

### 4. Document Thoroughly
Use docstrings with clear descriptions of:
- What the tool does
- All parameters (args)
- Return value structure
- Any side effects

### 5. Use the @tool Decorator
```python
from strands import tool

@tool
def my_tool(...):
    """..."""
    pass
```

## Tool Types

### Query Tools (Read-Only)
Tools that fetch information without side effects:
```python
@tool
def get_stock_price(symbol: str) -> dict:
    """Get current stock price (read-only)."""
    pass
```

### Action Tools (Side Effects)
Tools that perform actions or modify state:
```python
@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email (has side effects)."""
    pass
```

Mark action tools clearly in the docstring!

## Loading External APIs

If your tool needs API keys or external services:

```python
import os
from strands import tool

@tool
def my_api_tool(query: str) -> dict:
    """Use external API."""
    api_key = os.getenv("MY_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "MY_API_KEY not set in environment"
        }

    # Use API...
    return {"success": True, "result": ...}
```

Set environment variables in your `.env` file.

## Examples of Custom Tools

### Web Scraping
```python
@tool
def scrape_website(url: str) -> dict:
    """Scrape content from a website."""
    pass
```

### Data Processing
```python
@tool
def analyze_csv(file_path: str) -> dict:
    """Analyze CSV file and return statistics."""
    pass
```

### Integration with External Services
```python
@tool
def create_jira_ticket(title: str, description: str) -> dict:
    """Create a Jira ticket."""
    pass
```

### Custom Calculations
```python
@tool
def calculate_mortgage(principal: float, rate: float, years: int) -> dict:
    """Calculate monthly mortgage payment."""
    pass
```

## Testing Your Tools

Test your tools manually:

```python
# In Python REPL or test file
from tools.custom.my_tool import my_tool

result = my_tool("test input")
print(result)
```

## Getting Help

- See `example.py` for a complete template
- Check `tools/core/` for examples of well-structured tools
- Read the strands documentation: https://docs.strands.ai/

## Tool Discovery

The system automatically discovers tools in this directory. No registration needed!

Just ensure:
- Your file is a `.py` file in `tools/custom/`
- Your function has the `@tool` decorator
- The file is importable (no syntax errors)

Happy tool building! 🛠️
