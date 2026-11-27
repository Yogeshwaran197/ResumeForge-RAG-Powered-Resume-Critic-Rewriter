from cohere.types import chat_search_results_event
from rag import hybrid_search



if __name__ == "__main__":
    query = "what is name of the candidate?"
    retriever = hybrid_search()

    results = retriever.invoke(query)
    for r in results:
        print(r.page_content)