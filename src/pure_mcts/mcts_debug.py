import logging
import graphviz
from node import Node

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_tree(node: Node, depth: int = 0) -> None:
    """Recursively logs the tree structure."""
    if node is None:
        return
    logging.debug(f"{'  ' * depth}Node: {node.state}, Visits: {node.visits}, Value: {node.value}")
    for child in node.children:
        log_tree(child, depth + 1)

def visualize_tree(node: Node, graph=None) -> None:
    """Visualizes the tree structure using graphviz."""
    if graph is None:
        graph = graphviz.Digraph()

    graph.node(str(id(node)), f'State: {node.state}\nVisits: {node.visits}\nValue: {node.value}')
    
    for child in node.children:
        graph.edge(str(id(node)), str(id(child)))
        visualize_tree(child, graph)

    return graph

def save_tree_visualization(root: Node, filename: str = 'mcts_tree') -> None:
    """Saves the tree visualization to a file."""
    graph = visualize_tree(root)
    graph.render(filename, format='png', cleanup=True)  # Save the tree visualization
