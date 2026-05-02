"""W21 Agent-Tools — Re-Exports + Registry."""
from .web import web_search, web_fetch
from .spotify import spotify_top_artists, spotify_play
from .mood import get_mood

TOOL_REGISTRY = {
    "web_search": web_search,
    "web_fetch": web_fetch,
    "spotify_top_artists": spotify_top_artists,
    "spotify_play": spotify_play,
    "get_mood": get_mood,
}
