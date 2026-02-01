#!/usr/bin/env bash
# publish_config_to_new_repo.sh
# Copies selected config files into a temporary folder, initializes a new git repo and pushes to remote.
# Usage: ./scripts/publish_config_to_new_repo.sh https://github.com/your/repo.git

set -euo pipefail
REMOTE="${1:-https://github.com/znoux46/SEP490.8-WikiChatbot-RAGModel.git}"
TMP_DIR="$(pwd)/temp_config"
FILES=(docker-compose.yml Dockerfile init_db.sql migrations DOCKER_DEPLOYMENT.md README.md .github)

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "Copying files to $TMP_DIR"
for f in "${FILES[@]}"; do
  if [ -e "$f" ]; then
    echo " - $f"
    cp -r "$f" "$TMP_DIR/"
  else
    echo " - (missing) $f"
  fi
done

cat > "$TMP_DIR/.gitignore" <<'GITIGNORE'
.env
.env.*
GITIGNORE

cd "$TMP_DIR"
git init
git add .
git commit -m "Publish selected config files (no env)"
git remote add origin "$REMOTE"
git branch -M main
git push -u origin main --force

echo "Done. Temporary folder: $TMP_DIR"