FROM golang:1.22 AS build
WORKDIR /src
COPY . .
RUN go build -o /out/api .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/api /app/api
EXPOSE 8080
CMD ["/app/api"]
