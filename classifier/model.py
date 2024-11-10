from pydantic import BaseModel
from typing import Optional

# Define Rewrite model for structured validation
class New_Requirement(BaseModel):
    subdepartment: str
    new_requirement: str

# Define Rewrite model for structured validation
class Rewrite(BaseModel):
    example: list[New_Requirement]

# Define Classification model for structured validation
class Classification(BaseModel):
    classification: list[str]
    rewrite: Optional[Rewrite] = None
