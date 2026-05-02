"""W21 Agent-Tools — Re-Exports + Registry."""
from .web import web_search, web_fetch
from .spotify import (
    spotify_top_artists,
    spotify_top_tracks,
    spotify_play,
    spotify_pause,
    spotify_next,
    spotify_prev,
    spotify_volume,
    spotify_search,
    spotify_now_playing,
    spotify_recommend,
    spotify_play_genre,
)
from .mood import get_mood
from .hardware import (
    ptz_pan,
    led_set,
    camera_snapshot,
    ptz_tilt,
    thermal_set_tension_pwm,
    get_face_id,
    get_npu_status,
)
from .system import (
    get_audit_state,
    moloch_status_summary,
    read_memory,
    tts_say,
)
from .browser import (
    browser_open,
    browser_click,
    browser_screenshot,
)

TOOL_REGISTRY = {
    "web_search": web_search,
    "web_fetch": web_fetch,
    "spotify_top_artists": spotify_top_artists,
    "spotify_top_tracks": spotify_top_tracks,
    "spotify_play": spotify_play,
    "spotify_pause": spotify_pause,
    "spotify_next": spotify_next,
    "spotify_prev": spotify_prev,
    "spotify_volume": spotify_volume,
    "spotify_search": spotify_search,
    "spotify_now_playing": spotify_now_playing,
    "spotify_recommend": spotify_recommend,
    "spotify_play_genre": spotify_play_genre,
    "get_mood": get_mood,
    "ptz_pan": ptz_pan,
    "led_set": led_set,
    "camera_snapshot": camera_snapshot,
    "ptz_tilt": ptz_tilt,
    "thermal_set_tension_pwm": thermal_set_tension_pwm,
    "get_face_id": get_face_id,
    "get_npu_status": get_npu_status,
    "get_audit_state": get_audit_state,
    "moloch_status_summary": moloch_status_summary,
    "read_memory": read_memory,
    "tts_say": tts_say,
    "browser_open": browser_open,
    "browser_click": browser_click,
    "browser_screenshot": browser_screenshot,
}
