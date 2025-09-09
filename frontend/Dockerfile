# Frontend Dev Dockerfile (Vite + React)
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer)
COPY package*.json ./
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Vite uses VITE_* env vars at runtime in dev mode
ENV VITE_API_BASE_URL=http://localhost:5000

# Expose dev server port
EXPOSE 5173

# Run vite in dev mode
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
