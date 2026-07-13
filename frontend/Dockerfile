# React (Vite) dev server
FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173
# --host so the container's Vite is reachable from the host browser
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
