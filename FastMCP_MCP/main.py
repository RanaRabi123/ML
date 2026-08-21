import random
from fastmcp import FastMCP

mcp = FastMCP(name= 'demo_server')

@mcp.tool
def rool_dice(n_dice : int = 1) -> list[int] :
    """Roll a dice and return a number """
    return [random.randint(1, 6) for _ in range(n_dice)]


@mcp.tool 
def add_numebr(a:float, b:float)-> float:
    """Add the numbers and return the result """
    return a+b

if __name__ == "__main__":
    mcp.run()
    