from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


build = GraphBuilder(StateGraph, START, END)
build.add_node("start", start_node)
build.add_node("end", end_node)
build.add_edge(START, "start")
build.add_edge("start", "end")
build.add_edge(END, END)

graph = build.compile()