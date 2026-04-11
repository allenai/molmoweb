#!/bin/bash

# Development server with Socket.IO support using uvicorn
exec \
    uvicorn \
    app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
