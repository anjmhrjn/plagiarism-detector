from fastapi import FastAPI

app = FastAPI(title="Plagiarism Detector", version="0.0.1")

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 while the service is up."""
    return {"status": "ok"}