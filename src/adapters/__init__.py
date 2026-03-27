# src/GraphContainer/adapters/__init__.py
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError
from .fastinsight import FastInsightAdapter, import_graph_from_fastinsight
from .expla_graphs import ExplaGraphsAdapter, import_graph_from_expla_graphs
from .freebasekg import FreebaseKGAdapter, import_graph_from_freebasekg
from .g_retriever import GRetrieverAdapter, import_graph_from_g_retriever
from .lightrag import LightRAGAdapter, import_graph_from_lightrag
from .hipporag import HippoRAGAdapter, import_graph_from_hipporag

import_graph_from_component_graph = import_graph_from_fastinsight
import_graph_from_attribute_bundle_graph = import_graph_from_lightrag
import_graph_from_topology_semantic_graph = import_graph_from_hipporag
import_graph_from_subgraph_union_graph = import_graph_from_g_retriever

__all__ = [
    "GraphAdapter",
    "GraphAdapterError",
    "UnsupportedSourceError",
    "FastInsightAdapter",
    "ExplaGraphsAdapter",
    "FreebaseKGAdapter",
    "GRetrieverAdapter",
    "LightRAGAdapter",
    "HippoRAGAdapter",
    "import_graph_from_component_graph",
    "import_graph_from_attribute_bundle_graph",
    "import_graph_from_topology_semantic_graph",
    "import_graph_from_subgraph_union_graph",
    "import_graph_from_fastinsight",
    "import_graph_from_expla_graphs",
    "import_graph_from_freebasekg",
    "import_graph_from_lightrag",
    "import_graph_from_hipporag",
    "import_graph_from_g_retriever",
]
