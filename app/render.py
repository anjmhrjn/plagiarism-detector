import html as _html

_ANSI_BOLD_YELLOW = "\033[33;1m"
_ANSI_RESET = "\033[0m"


def render_html(query_text: str, spans: list[dict]) -> str:
    """HTML fragment with matched spans wrapped in <mark>. Safe to embed in a page."""
    parts: list[str] = []
    cursor = 0
    for s in sorted(spans, key=lambda x: x["start"]):
        if s["start"] > cursor:
            parts.append(_html.escape(query_text[cursor : s["start"]]))
        parts.append(f"<mark>{_html.escape(query_text[s['start'] : s['end']])}</mark>")
        cursor = s["end"]
    if cursor < len(query_text):
        parts.append(_html.escape(query_text[cursor:]))
    return "".join(parts)


def render_terminal(query_text: str, spans: list[dict]) -> str:
    """Plain string with matched spans highlighted in bold yellow for terminal display."""
    parts: list[str] = []
    cursor = 0
    for s in sorted(spans, key=lambda x: x["start"]):
        if s["start"] > cursor:
            parts.append(query_text[cursor : s["start"]])
        parts.append(f"{_ANSI_BOLD_YELLOW}{query_text[s['start'] : s['end']]}{_ANSI_RESET}")
        cursor = s["end"]
    if cursor < len(query_text):
        parts.append(query_text[cursor:])
    return "".join(parts)
