"""
Data Processor Skill Injection

Variables in __exports__ are automatically injected into the
agent's runtime when the skill is activated. Scripts can access
these via runtime.retrieve("variable_name").
"""

from cave_agent.runtime import Variable

# Sales transactions to be processed
sales_data = [
    {"amount": 800, "category": "electronics", "region": "north"},
    {"amount": 250, "category": "clothing", "region": "south"},
    {"amount": 1200, "category": "furniture", "region": "north"},
    {"amount": 150, "category": "groceries", "region": "east"},
    {"amount": 950, "category": "electronics", "region": "west"},
    {"amount": 320, "category": "clothing", "region": "north"},
]

# Sales targets by region
regional_targets = {
    "north": 2000,
    "south": 1500,
    "east": 1000,
    "west": 1200,
}

__exports__ = [
    Variable(
        "sales_data",
        value=sales_data,
        description="List[Dict] of sales transactions with keys: amount, category, region"
    ),
    Variable(
        "regional_targets",
        value=regional_targets,
        description="Dict[str, float] mapping region (north, south, east, west) to sales target"
    ),
]
