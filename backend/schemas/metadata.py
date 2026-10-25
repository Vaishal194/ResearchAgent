from pydantic import BaseModel, Field
from typing import Optional, List


class ResearchMetadata(BaseModel):
    title: Optional[str] = None

    authors: List[str] = Field(default_factory=list)

    affiliations: List[str] = Field(default_factory=list)

    publication_year: Optional[int] = None

    doi: Optional[str] = None

    abstract: Optional[str] = None

    keywords: List[str] = Field(default_factory=list)

    research_domain: Optional[str] = None