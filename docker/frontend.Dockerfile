FROM node:18-alpine

WORKDIR /app

COPY frontend/package*.json /app/frontend/
WORKDIR /app/frontend
RUN npm install --silent

COPY frontend/ /app/frontend/

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
