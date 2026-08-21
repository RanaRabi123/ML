from fastmcp import FastMCP
from typing import List

# 1. Create the server instance. This name shows up in the client's tool list.
mcp = FastMCP("Calculator Tools")


# 2. Each function decorated with @mcp.tool() becomes a callable "tool".
#    - Type hints matter: FastMCP uses them to build the JSON schema
#      the LLM sees, so it knows what arguments to pass.
#    - The docstring matters just as much: it's what the LLM reads to
#      decide *when* to call this tool. Vague docstrings = misuse.

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the difference (a - b)."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


@mcp.tool()
def product(numbers: List[float]) -> float:
    """
    Multiply a list of numbers together and return the result.
    Example: product([2, 3, 4]) -> 24
    """
    result = 1.0
    for n in numbers:
        result *= n
    return result


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b and return the quotient (a / b). Raises an error if b is 0."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@mcp.tool()
def modulus(a: int, b: int) -> int:
    """Return the remainder of a divided by b (a % b). Raises an error if b is 0."""
    if b == 0:
        raise ValueError("Modulus by zero is not allowed.")
    return a % b


# 3. Entry point: starts the server over stdio so an MCP client can connect.
if __name__ == "__main__":
    mcp.run()