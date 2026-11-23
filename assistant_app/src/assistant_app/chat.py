import sys
import time

from .pipeline import HISTORY_TURNS, LLMError, SearchError, answer, load

LLM_DOWN = (
    "XƏTA: LLM cavab vermir (Ollama işləmir və ya vaxt bitdi). "
    "Ollama-nı işə salıb yenidən cəhd edin."
)


def show(text):
    print(text, end="", flush=True)


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    start = time.perf_counter()

    try:
        load()
        print(f"LLM hazırdır ({time.perf_counter() - start:.1f}s)")
    except LLMError:
        print(LLM_DOWN)

    history = []

    while True:
        try:
            question = input("\nSual ('q' = çıxış): ").strip()
        except EOFError:
            break

        if not sys.stdin.isatty():
            print(question)

        if question.lower() == "q":
            break

        if not question:
            continue

        def on_query(query):
            if query != question:
                print(f"Axtarış sorğusu: {query}")

            print()

        try:
            result = answer(question, history, on_query=on_query, on_token=show)
        except LLMError:
            print(f"\n{LLM_DOWN}")
            continue
        except SearchError:
            print("\nXƏTA: axtarış alınmadı, yenidən cəhd edin.")
            continue

        history.append({"question": question, "answer": result["answer"]})
        if HISTORY_TURNS > 0:
            del history[:-HISTORY_TURNS]
        else:
            history.clear()

        if result["show_sources"]:
            print("\n\nMənbələr:")

            for number, source in enumerate(result["sources"], start=1):
                print(f"{number}. {source['title']} - {source['url']}")
        else:
            print()

        timings = result["timings"]
        print(
            f"\nSorğu hazırlığı: {timings['translate']:.2f}s | "
            f"Axtarış: {timings['search']:.2f}s | LLM: {timings['llm']:.2f}s"
        )


if __name__ == "__main__":
    main()
