import requests
import time
import random
import nltk
import subprocess

from ollama import Client

from newspaper import Article

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Collapsible

hn_base = "https://hacker-news.firebaseio.com/v0"
ollama_url = "http://localhost:11434"

summary_model = "llama3:8b"


def get_article_context(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    return article.text


def get_best_articles(n: int = 20):
    ids = requests.get(f"{hn_base}/topstories.json").json()[:n]
    articles = []
    for id in ids:
        article_data = requests.get(f"{hn_base}/item/{id}.json").json()
        try:
            articles.append(
                (
                    get_article_context(article_data["url"]),
                    article_data["title"],
                    article_data["by"],
                )
            )
        except:
            articles.append("Ignore this mesage since context failed with no reply")
        time.sleep(1 + random.random())
    print("Obtained best articles.")

    return articles


def get_summaries(articles: list = []):
    ollama_running = False

    try:
        r = requests.get(ollama_url)
        ollama_running = r.status_code == 200
    except requests.exceptions.RequestException:
        subprocess.Popen(["ollama", "serve"])
        time.sleep(3)
        ollama_running = True

    if not ollama_running:
        raise Exception("FAILURE running ollama!\n")

    client = Client(
        host="http://localhost:11434",
    )

    client.pull(summary_model)
    print(f"Pulled model {summary_model} from Ollama")

    summaries = []

    for article in articles:
        resp = client.chat(
            model=summary_model,
            messages=[
                {
                    "role": "user",
                    "content": f""""
                    Summarize the following content in 5 sentences: {article[0]}
                    - Do not add any other text other than the summary, make sure to be maximally concise
                    """,
                }
            ],
            think=False,
        )
        summaries.append(resp["message"]["content"])

        print(f"Completed summary #{len(summaries)}")
    return summaries


class HNApp(App):
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Footer()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""

        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


if __name__ == "__main__":
    nltk.download("punkt_tab")
    n = 3
    articles = get_best_articles(n)
    summaries = get_summaries(articles)
    print(summaries)
