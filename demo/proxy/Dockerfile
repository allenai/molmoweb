FROM nginx:1.17.0-alpine

COPY nginx.conf /etc/nginx/nginx.conf

ARG CONF_FILE=local.conf
COPY $CONF_FILE /etc/nginx/conf.d/default.conf

