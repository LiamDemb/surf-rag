__all__ = ["DenseRetriever", "GraphRetriever"]


def __getattr__(name: str):
    if name == "DenseRetriever":
        from surf_rag.strategies.dense import DenseRetriever

        return DenseRetriever
    if name == "GraphRetriever":
        from surf_rag.strategies.graph import GraphRetriever

        return GraphRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))
