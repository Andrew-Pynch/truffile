#!/usr/bin/env bash
# Deploy the ArXiv app to Truffle device.
# Syncs credentials, validates, deletes old instance, deploys fresh.
set -e

cd "$(dirname "$0")/arxiv"

echo "=== Syncing AlphaXiv credentials ==="
python sync_creds.py

echo ""
echo "=== Validating ==="
truffile validate .

echo ""
echo "=== Deleting old instance ==="
truffile delete all

echo ""
echo "=== Deploying ==="
truffile deploy .

echo ""
echo "=== Listing apps ==="
truffile list apps
