#!/bin/bash
# filepath: /Users/devenshah/repos/model-distribution-server/startup.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Model Distribution Server...${NC}"

# Function to check if a service is ready
check_service() {
    local url=$1
    local service_name=$2
    local max_attempts=60
    local attempt=1

    echo -e "${YELLOW}⏳ Waiting for $service_name to be ready at $url${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is ready!${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}   Attempt $attempt/$max_attempts - $service_name not ready yet...${NC}"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $service_name failed to start within timeout${NC}"
    return 1
}

# Function to start server in new terminal
start_server() {
    local command=$1
    local server_name=$2
    local port=$3
    
    echo -e "${BLUE}🖥️  Starting $server_name on port $port${NC}"
    
    # For macOS (Terminal.app)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e "tell app \"Terminal\" to do script \"cd '$PWD' && source .venv/bin/activate && $command\""
    # For Linux (gnome-terminal)
    elif command -v gnome-terminal &> /dev/null; then
        gnome-terminal --working-directory="$PWD" -- bash -c "source .venv/bin/activate && $command; exec bash"
    # For Linux (xterm fallback)
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$PWD' && source .venv/bin/activate && $command; bash" &
    # For WSL/Windows Terminal
    elif command -v wt &> /dev/null; then
        wt -d "$PWD" bash -c "source .venv/bin/activate && $command"
    else
        echo -e "${YELLOW}⚠️  Could not detect terminal. Please manually run: $command${NC}"
    fi
}

# Step 1: Start Docker services
echo -e "${BLUE}📦 Starting Docker services...${NC}"
docker compose down 2>/dev/null || true  # Clean up any existing containers
docker compose up -d

echo -e "${GREEN}✅ Docker containers started${NC}"

# Step 2: Wait for services to be ready
check_service "http://localhost:5001" "MLflow Server"
check_service "http://localhost:9001/login" "MinIO Console"

# Check if mlflow bucket exists and create if needed
echo -e "${YELLOW}🪣 Checking MinIO bucket...${NC}"
sleep 2  # Give MinIO a moment to fully initialize
if ! docker compose exec -T minio mc alias set local http://localhost:9000 minio minio123 2>/dev/null; then
    echo -e "${RED}❌ Failed to connect to MinIO${NC}"
    exit 1
fi

if ! docker compose exec -T minio mc ls local/mlflow 2>/dev/null; then
    echo -e "${YELLOW}🔨 Creating mlflow bucket...${NC}"
    docker compose exec -T minio mc mb local/mlflow
    echo -e "${GREEN}✅ mlflow bucket created${NC}"
else
    echo -e "${GREEN}✅ mlflow bucket already exists${NC}"
fi

# Step 3: Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}🐍 Virtual environment not found. Creating...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r mds_train_server/requirements.txt
    pip install -r mds_inference_server/requirements.txt
    echo -e "${GREEN}✅ Virtual environment created and dependencies installed${NC}"
fi

# Step 4: Start application servers in new terminals
echo -e "${BLUE}🖥️  Starting application servers...${NC}"
sleep 1

start_server "uvicorn mds_train_server.app:app --reload --port 8001" "Training Server" "8001"
sleep 2
start_server "uvicorn mds_inference_server.app:app --reload --port 8002" "Inference Server" "8002"

# Final status
echo -e "${GREEN}🎉 Model Distribution Server is starting up!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}📊 MLflow UI:      http://localhost:5001${NC}"
echo -e "${GREEN}🗄️  MinIO Console:  http://localhost:9001 (minio/minio123)${NC}"
echo -e "${GREEN}🔧 Training API:   http://localhost:8001${NC}"
echo -e "${GREEN}🔮 Inference API:  http://localhost:8002${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"

echo -e "${BLUE}💡 Test training with:${NC}"
echo -e "${YELLOW}curl -X POST -F \"file=@sample_data/time_series_weather.csv\" -F \"model_name=rfr\" http://localhost:8001/train${NC}"

echo -e "${BLUE}💡 View logs with:${NC}"
echo -e "${YELLOW}docker compose logs -f${NC}"

open "http://localhost:5001"
open "http://localhost:9001"