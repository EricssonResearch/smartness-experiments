#!/bin/bash
SEED_HOST=$1
echo "Waiting for $SEED_HOST to become available..."

while ! busybox nc -z "$SEED_HOST" 7000; do
  sleep 1
done

echo "Seed $SEED_HOST available. Starting Cassandra..."
exec docker-entrypoint.sh cassandra