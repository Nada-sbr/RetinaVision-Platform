# Stage 1: Build the React application
FROM node:20-alpine AS build-stage

WORKDIR /app

# Copy dependency configs and install
COPY frontend/package*.json ./
RUN npm ci

# Copy source code and build assets
COPY frontend/ ./

# Build production bundle
RUN npm run build

# Stage 2: Serve using Nginx
FROM nginx:stable-alpine AS production-stage

# Copy custom Nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy static assets from build stage to Nginx html directory
COPY --from=build-stage /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
