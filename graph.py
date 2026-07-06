from trial import build_graph

app = build_graph()

png = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png)

print("Saved graph to graph.png")