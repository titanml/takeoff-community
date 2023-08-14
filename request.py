"""Demo of streaming predictions from the server."""
# ───────────────────────────────────────────────────── imports ────────────────────────────────────────────────────── #

import requests


def stream_predictions(text):
    """Stream predictions from the server.

    Args:
        text (str): prompt text
    """
    response = requests.post("http://localhost:8000/generate_stream", json={"text": text}, stream=True)

    if response.encoding is None:
        response.encoding = "utf-8"

    for text in response.iter_content(chunk_size=1, decode_unicode=True):
        if text:
            print(text, end="", flush=True)


def generate_predictions(text):
    """Generate predictions from the server.

    Args:
        text (str): prompt text
    """
    response = requests.post("http://localhost:8000/generate", json={"text": text})

    if "message" in response.json():
        print(response.json()["message"])


if __name__ == "__main__":
    stream_predictions("List three things you can do in London")
    generate_predictions("List three things you can do in London")
