#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export KAGGLE_USERNAME="$(grep '^KAGGLE_USERNAME' "$ROOT/.env" | cut -d= -f2- | tr -d " '\"")"
export KAGGLE_KEY="$(grep '^KAGGLE_KEY' "$ROOT/.env" | cut -d= -f2- | tr -d " '\"")"
cd "${KG_CWD:-/tmp}" && exec uvx -q --with six kaggle "$@"
