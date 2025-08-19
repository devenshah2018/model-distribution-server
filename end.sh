#!/bin/bash
# filepath: /Users/devenshah/repos/model-distribution-server/shutdown.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Shutting down Model Distribution Server...${NC}"

# Function to kill processes by port
kill_by_port() {
    local port=$1
    local service_name=$2
    
    echo -e "${YELLOW}🔍 Looking for $service_name on port $port...${NC}"
    
    # Find process using the port
    local pid=$(lsof -ti:$port 2>/dev/null || echo "")
    
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}⚠️  Killing $service_name (PID: $pid)${NC}"
        kill -TERM $pid 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            echo -e "${RED}🔥 Force killing $service_name (PID: $pid)${NC}"
            kill -KILL $pid 2>/dev/null || true
        fi
        
        echo -e "${GREEN}✅ $service_name stopped${NC}"
    else
        echo -e "${GREEN}✅ $service_name not running${NC}"
    fi
}

# Function to kill uvicorn processes by name
kill_uvicorn_processes() {
    echo -e "${YELLOW}🔍 Looking for uvicorn processes...${NC}"
    
    # Kill uvicorn processes for our specific apps
    pkill -f "mds_train_server.app:app" 2>/dev/null || true
    pkill -f "mds_inference_server.app:app" 2>/dev/null || true
    
    # Wait a moment
    sleep 2
    
    # Force kill any remaining uvicorn processes on our ports
    kill_by_port "8001" "Training Server"
    kill_by_port "8002" "Inference Server"
}

# Step 1: Kill application servers
echo -e "${BLUE}🖥️  Stopping application servers...${NC}"
kill_uvicorn_processes

# Step 2: Stop Docker services
echo -e "${BLUE}📦 Stopping Docker services...${NC}"
if docker compose ps -q > /dev/null 2>&1; then
    docker compose down
    echo -e "${GREEN}✅ Docker containers stopped${NC}"
else
    echo -e "${GREEN}✅ No Docker containers to stop${NC}"
fi

# Step 3: Optional cleanup (commented out by default)
# Uncomment these lines if you want to clean up Docker resources
# echo -e "${YELLOW}🧹 Cleaning up Docker resources...${NC}"
# docker compose down -v  # Remove volumes
# docker system prune -f  # Remove unused containers, networks, images

# Final status
echo -e "${GREEN}🎉 Model Distribution Server shut down complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Training Server stopped${NC}"
echo -e "${GREEN}✅ Inference Server stopped${NC}"
echo -e "${GREEN}✅ Docker services stopped${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"

echo -e "${BLUE}💡 To restart, run:${NC}"
echo -e "${YELLOW}./start.sh${NC}"