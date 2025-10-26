#!/bin/bash

echo "🚀 Starting Ad Intelligence Web App..."
echo ""

# Check if backend files exist
if [ ! -f "backend/app.py" ]; then
    echo "❌ Error: backend/app.py not found!"
    echo "Please make sure you're in the correct directory"
    exit 1
fi

# Check if .env.local exists
if [ ! -f "backend/.env.local" ]; then
    echo "⚠️  Warning: backend/.env.local not found!"
    echo "Creating template..."
    echo "LAVA_BASE_URL=https://api.lavapayments.com/v1
LAVA_FORWARD_TOKEN=your-token-here" > backend/.env.local
    echo ""
    echo "📝 Please edit backend/.env.local with your Lava credentials"
    echo "Then run this script again"
    exit 1
fi

# Start backend in background
echo "🔧 Starting backend server..."
cd backend
python3 app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both servers started!"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "🌐 Open your browser to: http://localhost:3000"
echo ""
echo "To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Wait for user interrupt
wait