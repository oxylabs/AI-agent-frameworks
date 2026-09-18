"""The output format every agent must produce."""

from pydantic import BaseModel, Field

KEY_POINTS = 4
MAX_WORDS = 50


class Article(BaseModel):
    """One page in the digest."""

    title: str = Field(
        description="The headline shown on the scraped page."
    )
    source: str = Field(
        description="The publisher of the page."
    )
    link: str = Field(
        description="The page URL, copied from the search result."
    )
    key_points: list[str] = Field(
        min_length=KEY_POINTS,
        max_length=KEY_POINTS,
        description=(
            f"Exactly {KEY_POINTS} bullet points on which stocks the page "
            f"picks and the reasons it gives, each under {MAX_WORDS} "
            "words."
        ),
    )


class Digest(BaseModel):
    """Every page, written up."""

    articles: list[Article]
