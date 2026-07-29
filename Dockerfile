FROM nginx:alpine

# 1. Copy everything to the root folder (for index.html)
COPY . /usr/share/nginx/html/

# 2. Duplicate your map files and CSVs into a 'map' subfolder so the browser paths match
RUN mkdir -p /usr/share/nginx/html/map
COPY *.html *.csv *.geojson /usr/share/nginx/html/map/

EXPOSE 8443
