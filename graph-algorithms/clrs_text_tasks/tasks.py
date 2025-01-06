
import random
import networkx as nx
import numpy as np
from clrs_text_tasks import graph_text_encoder


class GraphTask:
	"""The parent class for all the graph tasks."""
	answer_prompt = "Answer: {answer}"
	chain_of_thought_promt = "Steps: \n{steps}"

	def __init__(self):
		self.name = 'default'
		self.problem_description = 'default'
		self.algorithm_description = 'default'
		self.question_prompt = 'default'
		self.answer_prompt = 'default'
		self.chain_of_thought_promt = 'default'

	def prepare_examples(
		self,
		graphs,
		encoding_method,
	):
		raise NotImplementedError()

class BellmanFordTask(GraphTask):
	"""The Bellman-Ford task."""

	def __init__(self, add_cot=False):
		self.name = 'bellman_ford'
		self.problem_description = "In a single-source shortest-paths problem, we are given a weighted directed graph and a source node. We aim to find the shortest path starting from source node to every other node. For each node, each node has a pointer to its predecessor along the optimal path from the source. By convention, the source node points to itself. "
		self.algorithm_description = "The Bellman-Ford algorithm starts by initializing the distance to the source vertex as 0 and all other vertices as infinity. It then iteratively relaxes each edge in the graph V-1 times where V is the number of vertices. Relaxing an edge involves checking whether the distance to the destination vertex can be reduced by taking the edge and, if so, updating the distance and the predecessor. After V-1 iterations, the algorithm performs an additional pass to check for negative-weight cycles. If any edge can still be relaxed, it indicates the presence of a negative-weight cycle, which is reported. Otherwise, the shortest paths from the source to all vertices are finalized."

		self.question_prompt = "Question: Return the the predecessor nodes of all nodes in the shortest path to node {source} in alphabetical order."
		self.answer_prompt = "{answer}"
		self.chain_of_thought_promt = "Steps: \n{steps}"
		self.add_cot = add_cot
	
	def get_answer(self, data):
		answer = list(data.edge_index.T[data.pi==1])
		answer = sorted(answer, key=lambda x: x[0])
		answer = [str(x[1].item()) for x in answer]
		answer = ", ".join(answer)
		return answer

	def get_intermediate_results(self, data):
		hints = data.pi_h
		intermediate_results = []
		for i in range(hints.shape[1]):
			answer = list(data.edge_index.T[data.pi_h[:, i]==1])
			if len(answer) < data.num_nodes: break
			answer = sorted(answer, key=lambda x: x[0])
			answer = [str(x[1].item()) for x in answer]
			answer = ", ".join(answer)
			intermediate_results.append(answer)
		return intermediate_results # exclude the last step since it is the answer
	
	def prepare_examples(
		self,
		dataset,
		encoding_method="incident",
	):
		examples = []
		name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
		for ind, data in enumerate(dataset):
			graph = nx.from_edgelist(data.edge_index.T.numpy())
			graph_encoding = graph_text_encoder.encode_graph(graph, encoding_method)
			source = data.s.nonzero().item()
			source = name_dict[source]
			question = self.problem_description + "\n" + self.algorithm_description + "\n"
			question += graph_encoding + self.question_prompt.format(source=source) 
			if self.add_cot:
				question += "\n" + self.chain_of_thought_promt.format(steps="\n".join(self.get_intermediate_results(data)))
			question += "\nAnswer: "
			answer = self.get_answer(data)
			examples.append({
				'id': str(ind),
				'question': question,
				'answer': answer,
			})
		return examples