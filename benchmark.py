"""Local sequential API timing; not a concurrent load test."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from app.main import create_app

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as d, TestClient(create_app('artifacts/recommender.joblib',str(Path(d)/'events.db'))) as client:
        timings=[]
        for user in range(1,101):
            start=time.perf_counter();response=client.get(f'/recommendations/{user}');response.raise_for_status();timings.append((time.perf_counter()-start)*1000)
        report={'requests':100,'p50_ms':float(np.percentile(timings,50)),'p95_ms':float(np.percentile(timings,95)),'measurement':'Sequential local in-process FastAPI TestClient requests including SQLite history lookup; not external network or concurrency load testing'}
        Path('reports/latency.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
