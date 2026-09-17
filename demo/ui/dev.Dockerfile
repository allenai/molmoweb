FROM node:18-alpine

WORKDIR /ui

COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

COPY . .

ENTRYPOINT [ "yarn" ]
CMD [ "dev" ]
