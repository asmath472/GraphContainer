from .core import SearchableGraphContainer, SimpleGraphContainer
from typing import Optional

# 이렇게 까지 필요한지는 잘 모르겠다.
def container_or_new(container: Optional[SimpleGraphContainer]) -> SearchableGraphContainer:
    if container is None:
        return SearchableGraphContainer()
    if isinstance(container, SearchableGraphContainer):
        return container
    graph = SearchableGraphContainer()
    graph.nodes = dict(container.nodes)
    graph.edges = list(container.edges)
    graph._adj = {node_id: list(indices) for node_id, indices in container._adj.items()}
    return graph
