# agents/web_agent.py

from urllib.parse import quote_plus

from duckduckgo_search import DDGS

from utils import open_with, speak, log_agent, iso_now


class WebAgent:
    def search(self, cmd):
        norm = (cmd.get("normalized") or {}) or {}
        q = norm.get("search_query")

        if not q:
            q = cmd.get("original_text") or "web search"
            for w in ["search for", "search", "look up", "find", "google", "web search"]:
                q = q.replace(w, "")
            q = q.strip()

        if not q:
            q = "latest news"

        url = f"https://www.google.com/search?q={quote_plus(q)}"
        print(f"(web) Searching: {q}")
        if open_with(["firefox", "google-chrome", "chromium-browser", "xdg-open"], [url]):
            speak("Opened browser with search results.")
        else:
            speak("Could not open browser, printing top results.")

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=5):
                    results.append(
                        {
                            "title": r.get("title"),
                            "href": r.get("href"),
                            "snippet": r.get("body"),
                        }
                    )
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']}\n {r['href']}\n {r['snippet']}\n")
            log_agent(
                {
                    "agent": "web",
                    "event": "results",
                    "query": q,
                    "results": results,
                    "ts": iso_now(),
                }
            )
        except Exception as e:
            print(f"(web) search failed: {e}")

