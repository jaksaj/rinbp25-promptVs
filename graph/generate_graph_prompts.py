import random
import json
import argparse
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set

class GraphPromptGenerator:
    def __init__(self, num_graphs: int = 10, num_questions: int = 1, max_nodes: int = 10):
        self.num_graphs = num_graphs
        self.num_questions = num_questions
        self.max_nodes = min(max_nodes, 26)  # Limit to alphabet size
        self.node_pool = [chr(ord('A') + i) for i in range(26)]  # Generate A-Z
    
    def generate_random_graph(self, min_nodes: int = 4, max_nodes: int = None) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Generate a random directed graph with given node range."""
        if max_nodes is None:
            max_nodes = min(self.max_nodes, 7)  # Use smaller default for readability
        else:
            max_nodes = min(max_nodes, self.max_nodes)
        
        min_nodes = min(min_nodes, max_nodes)
        num_nodes = random.randint(min_nodes, max_nodes)
        nodes = self.node_pool[:num_nodes]
        
        # Generate edges - ensure some connectivity
        edges = []
        edge_set = set()
        
        # Create a basic connected structure
        for i in range(len(nodes) - 1):
            from_node = nodes[i]
            to_node = nodes[i + 1]
            edges.append((from_node, to_node))
            edge_set.add((from_node, to_node))
        
        # Add random additional edges
        num_additional_edges = random.randint(1, len(nodes))
        for _ in range(num_additional_edges):
            from_node = random.choice(nodes)
            to_node = random.choice(nodes)
            if from_node != to_node and (from_node, to_node) not in edge_set:
                edges.append((from_node, to_node))
                edge_set.add((from_node, to_node))
        
        return nodes, edges
    
    def build_adjacency_list(self, edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        graph = defaultdict(list)
        for from_node, to_node in edges:
            graph[from_node].append(to_node)
        return dict(graph)
    
    def bfs_at_depth(self, graph: Dict[str, List[str]], start: str, depth: int) -> List[str]:
        """Perform BFS and return nodes at specific depth."""
        if depth == 0:
            return []
        
        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)
        nodes_at_depth = []
        
        while queue:
            node, current_depth = queue.popleft()
            
            if current_depth == depth:
                nodes_at_depth.append(node)
            elif current_depth < depth:
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
        
        return sorted(nodes_at_depth)
    
    def find_parents(self, edges: List[Tuple[str, str]], target: str) -> List[str]:
        """Find all nodes that have edges pointing to the target node."""
        parents = []
        for from_node, to_node in edges:
            if to_node == target and from_node not in parents:
                parents.append(from_node)
        return sorted(parents)
    
    def find_reachable_nodes(self, graph: Dict[str, List[str]], start: str) -> List[str]:
        """Find all nodes reachable from start node."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Remove start node from result
        visited.discard(start)
        return sorted(list(visited))
    
    def generate_question(self, nodes: List[str], edges: List[Tuple[str, str]], question_type: str) -> Dict[str, str]:
        """Generate a specific type of question for the graph."""
        graph = self.build_adjacency_list(edges)
        edges_text = "\n".join([f"{from_node} -> {to_node}" for from_node, to_node in edges])
        manual_instruction = ("IMPORTANT: Do NOT provide code. Instead, provide the exact answer (the list of nodes or result) as requested. Do not show or describe any code.")

        if question_type == "bfs":
            start_node = random.choice(nodes)
            depth = random.randint(1, 3)
            solution = self.bfs_at_depth(graph, start_node, depth)

            return {
                "name": f"Graph BFS from {start_node}",
                "description": "Perform breadth-first search on a directed graph",
                "content": f"You will be given a list of directed edges and an operation to perform.\n"
                          f"For BFS, return only the nodes reached at the specified depth (not the starting node).\n"
                          f"For parents, return only nodes with edges pointing to the given node (not the node itself).\n"
                          f"The graph has the following edges:\n{edges_text}\n"
                          f"Operation:\nPerform a BFS from node {start_node} with depth {depth}.\n"
                          f"{manual_instruction}",
                "expected_solution": str(solution),
                "tags": ["graph", "bfs", "algorithms"]
            }

        elif question_type == "parents":
            target_node = random.choice(nodes)
            solution = self.find_parents(edges, target_node)

            return {
                "name": f"Find parents of {target_node}",
                "description": "Find all parent nodes of a given node",
                "content": f"You will be given a list of directed edges and an operation to perform.\n"
                          f"For BFS, return only the nodes reached at the specified depth (not the starting node).\n"
                          f"For parents, return only nodes with edges pointing to the given node (not the node itself).\n"
                          f"The graph has the following edges:\n{edges_text}\n"
                          f"Operation:\nFind all parent nodes of {target_node}.\n"
                          f"{manual_instruction}",
                "expected_solution": str(solution),
                "tags": ["graph", "parents", "algorithms"]
            }

        elif question_type == "reachable":
            start_node = random.choice(nodes)
            solution = self.find_reachable_nodes(graph, start_node)

            return {
                "name": f"Reachable nodes from {start_node}",
                "description": "Find all nodes reachable from a given starting node",
                "content": f"You will be given a list of directed edges and an operation to perform.\n"
                          f"For reachable, return all nodes that can be reached from the starting node (not the starting node itself).\n"
                          f"The graph has the following edges:\n{edges_text}\n"
                          f"Operation:\nFind all nodes reachable from {start_node}.\n"
                          f"{manual_instruction}",
                "expected_solution": str(solution),
                "tags": ["graph", "reachable", "algorithms"]
            }
    
    def generate_prompts_config(self) -> Dict:
        """Generate the complete prompts configuration."""
        all_prompts = []
        question_types = ["bfs", "parents", "reachable"]
        
        for graph_idx in range(self.num_graphs):
            nodes, edges = self.generate_random_graph()
            
            for question_idx in range(self.num_questions):
                question_type = question_types[question_idx % len(question_types)]
                prompt = self.generate_question(nodes, edges, question_type)
                all_prompts.append(prompt)
        
        config = {
            "prompt_group": {
                "name": "Generated Graph Test Prompts",
                "description": f"Auto-generated graph prompts with {self.num_graphs} graphs and {self.num_questions} questions each",
                "tags": ["graph", "algorithms", "generated", "testing"]
            },
            "prompts": all_prompts
        }
        
        return config

def main():
    parser = argparse.ArgumentParser(description="Generate graph prompts for testing")
    parser.add_argument("--graphs", type=int, default=2, help="Number of graphs to generate (default: 2)")
    parser.add_argument("--questions", type=int, default=5, help="Number of questions per graph (default: 3)")
    parser.add_argument("--max-nodes", type=int, default=10, help="Maximum number of nodes per graph (default: 10, max: 26)")
    parser.add_argument("--output", type=str, default="generated_graph_prompts.json", help="Output file name")
    
    args = parser.parse_args()
    
    generator = GraphPromptGenerator(args.graphs, args.questions, args.max_nodes)
    config = generator.generate_prompts_config()
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(config['prompts'])} prompts in {args.output}")
    print(f"Graphs: {args.graphs}, Questions per graph: {args.questions}")

if __name__ == "__main__":
    main()