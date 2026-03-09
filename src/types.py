# src/GraphContainer/types.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class NodeRecord(BaseModel):
    id: str
    type: str = "Entity"  # Entity, Chunk, Document 등
    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EdgeRecord(BaseModel):
    source: str
    target: str
    relation: str = "RELATED"
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)