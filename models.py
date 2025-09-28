from pydantic import BaseModel
from typing import List, Optional


class Coordinates(BaseModel):
    lat: float
    lon: float


class InputData(BaseModel):
    img_base64: str
    coordinates: Coordinates | None = None


class TreeInfo(BaseModel):
    species: str
    trunk_rot: bool
    hollows: bool
    cracks: bool
    trunk_damage: bool
    crown_damage: bool
    fruiting_bodies: bool       | None
    diseases: List[str]         | None = None
    dry_branches_percent: float | None = None
    other: str                  | None = None
    description: str


class OutputData(BaseModel):
    img_marked_base64: str
    trees: List[TreeInfo]
