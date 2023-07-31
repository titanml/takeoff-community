#!/bin/bash
# Start FastAPI server. If the RELOAD environment variable is set, add "--reload"
# Command without reload: uvicorn app.main:app --host '0.0.0.0' --port '80'
if [ -n "$RELOAD" ]; then
    echo "Serving on ${HOST_URL:-http://localhost:8000} with reload"
    uvicorn app.main:app --host '0.0.0.0' --port '80' --reload
else
    uvicorn app.main:app --host '0.0.0.0' --port '80'
fi
