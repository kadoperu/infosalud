"""
Client para interactuar con el pipeline de LangGraph.
"""

from langgraph.client import Client

class LangGraphClient:
    def __init__(self):
        self.client = Client()

    def get_state(self):
        return self.client.get_state()