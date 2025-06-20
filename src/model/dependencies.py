from pydantic import BaseModel, Field
from typing import List

class Dependencies(BaseModel):
    dependencies: List[str] = Field(
        ...,
        description="List of dependencies that needed to be install"
    )