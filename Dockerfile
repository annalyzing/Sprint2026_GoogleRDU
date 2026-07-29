FROM nginx:alpine

# 1. Copy everything to the root folder (for index.html)
COPY . /usr/share/nginx/html/


EXPOSE 443
