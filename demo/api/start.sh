#!/bin/bash

# Start the FastAPI app with Socket.IO support using uvicorn
exec \
    uvicorn \
    app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --access-log
