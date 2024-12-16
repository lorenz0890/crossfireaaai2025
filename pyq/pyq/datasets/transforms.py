"""
By convention, masks should have true elements at positions where higher precision should be used
"""
from typing import Callable

from torch import bincount, cumsum, float, long
from torch_geometric.data import Batch
from torch_geometric.utils import degree

COMPUTED_MASK_NAME = "computed_mask"


def in_degree_mask(graph):
    edge_index, num_nodes = graph.edge_index[1], graph.num_nodes
    in_degree = degree(edge_index, num_nodes, dtype=long)
    in_degree_scaled_counts = bincount(in_degree)
    in_degree_scaled_counts = cumsum(in_degree_scaled_counts, dim=0, dtype=float)
    return in_degree_scaled_counts[in_degree]


class NormalizedDegree:
    """
    source: https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/datasets.py#L10
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class MaskMaker:
    """
    Preform an operation on the graph G(V, E), to be able to compute a mask
    """

    def __init__(self, per_graph: bool = True, process_graph_function: Callable = None):
        self.per_graph = per_graph
        self.process_graph_function = process_graph_function

    def __process_graph__(self, graph):
        if self.process_graph_function is None:
            raise NotImplementedError("provide the `process_graph_function` to the {}.".format(self.__class__.__name__))
        mask = self.process_graph_function(graph)
        setattr(graph, COMPUTED_MASK_NAME, mask)
        return graph

    def __call__(self, data):
        if self.per_graph and isinstance(data, Batch):
            processed_graphs = [self.__process_graph__(graph) for graph in data.to_data_list()]
            return Batch.from_data_list(processed_graphs)
        else:
            return self.__process_graph__(data)


class ProbabilityMaskMaker(MaskMaker):
    """
    Preform an operation on the graph G(V, E), to be able to compute a probabilistic mask
    """

    def __init__(
        self,
        low_probability: float,
        high_probability: float,
        per_graph: bool = True,
        process_graph_function: Callable = None,
    ):
        super(ProbabilityMaskMaker, self).__init__(per_graph=per_graph, process_graph_function=process_graph_function)
        self.low_probability = low_probability
        self.high_probability = high_probability

    def __process_graph__(self, graph):
        n = graph.num_nodes
        probability_range = (self.high_probability - self.low_probability) / n
        processed_graph = super(ProbabilityMaskMaker, self).__process_graph__(graph)
        mask = getattr(processed_graph, COMPUTED_MASK_NAME)
        probability_mask = mask * probability_range
        probability_mask += probability_mask * self.low_probability
        setattr(graph, COMPUTED_MASK_NAME, probability_mask)
        return graph
