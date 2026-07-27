#!/bin/bash
cd ~/Documents/capstone/HAIP/healthcare-project
docker system prune -f
docker compose up -d fastapi merged_app doctor_app streamlit agent_app
sleep 8
docker compose ps
fuser -k 5173/tcp 2>/dev/null
cd frontend && nohup npm run dev > /tmp/vite.log 2>&1 &
sleep 5 && tail -5 /tmp/vite.log
echo "=== HAIP READY === http://localhost:5173"


## for rnning this command 

##  cd ~/Documents/capstone/HAIP/healthcare-project && ./start.sh