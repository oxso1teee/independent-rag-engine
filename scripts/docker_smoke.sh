#!/usr/bin/env bash
# Smoke test of the Docker image: it must refuse to start without a token, run as an unprivileged
# user, answer health, require the token elsewhere, report healthy and stay under the size limit.
#
#   docker build -t rag-engine:smoke . && scripts/docker_smoke.sh rag-engine:smoke
set -euo pipefail

IMAGE="${1:?usage: docker_smoke.sh IMAGE}"
LIMIT_BYTES=$((400 * 1000 * 1000))
TOKEN="smoke-token-0123456789abcdef"
NAME="rag-smoke-$$"
PORT="${SMOKE_PORT:-18765}"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

size=$(docker image inspect "$IMAGE" --format '{{.Size}}')
echo "image size: $size bytes"
[ "$size" -le "$LIMIT_BYTES" ] || { echo "image exceeds 400 MB"; exit 1; }

if docker run --rm "$IMAGE" >/tmp/rag-smoke-notoken.log 2>&1; then
  echo "the server started without a token"; cat /tmp/rag-smoke-notoken.log; exit 1
fi
grep -q "token is required" /tmp/rag-smoke-notoken.log

docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8765" -e RAG_API_TOKEN="$TOKEN" "$IMAGE" >/dev/null
for _ in $(seq 1 30); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/v1/health" | grep -q '"status":"ok"'
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/stats")" = 401 ]
[ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${TOKEN}" \
    "http://127.0.0.1:${PORT}/v1/stats")" = 200 ]
[ "$(docker exec "$NAME" id -u)" = 10001 ]
echo "docker smoke test passed"
