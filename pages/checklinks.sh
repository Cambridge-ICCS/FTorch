#!/bin/bash
# Usage: ./checklinks.sh [FILES]
# Scans markdown files for broken links. Defaults to "*.md" if no FILES are provided.
# Only works for markdown links, and only absolute (http/https) links

# Exit on non-zero exit code for any calls.
set -e

# Directory or files to scan (can be passed as a command-line argument, defaults to "*.md")
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 [FILES]"
  echo "Scans markdown files for broken links."
  echo "  FILES: Optional. Specify files or patterns to scan. Defaults to \"*.md\"."
  exit 0
fi
FILES=${1:-"*.md"}

# grep -E regex to extract markdown links from markdown files - [text](link)
LINK_REGEX='\[[^]]+\]\((https?:\/\/[^[:space:]()]+(\([^[:space:]()]*\))?[^[:space:]()]*)\)'
# Regex to extract URL from markdown links
URL_REGEX='s/.*(\(http.*\)).*/\1/p'

# Function to check URL availability and log number of errors
check_url() {
  local url=$1
  echo "Checking URL: $url"
  status_code=$(curl --head --silent --fail --max-time 10 -L -o /dev/null -w "%{http_code}" "$url")
  if [[ "$status_code" -eq 000 ]]; then
    echo "⚠️ Warning: $url returned HTTP $status_code (possibly blocked or inaccessible)."
  elif [[ "$status_code" -ge 400 ]]; then
    echo "❌ $url is not reachable (HTTP $status_code)."
    return 1
  elif [[ "$status_code" -ge 300 ]]; then
    echo "⚠️ Warning: $url returned HTTP $status_code (redirect followed)."
  else
    echo "✅ $url is reachable (HTTP $status_code)."
  fi
  return 0
}

# Extract and check URLs
overall_errors=0
for file in $FILES; do
  echo "=============================================================================="
  echo "Scanning file: $file"
  echo "=============================================================================="
  file_errors=0
  if ! grep -Eo "$LINK_REGEX" "$file" > /dev/null; then
    echo "No matches found in $file"
  fi
  while IFS= read -r url; do
    if ! check_url "$url"; then
      file_errors=$((file_errors + 1))
      overall_errors=$((overall_errors + 1))
    fi
  done < <(grep -Eo "$LINK_REGEX" "$file" | sed -n "$URL_REGEX")
  if (( file_errors > 0 )); then
    echo "Found $file_errors broken links in $file."
  else
    echo "All links in $file are valid."
  fi
  echo -e "\n"
done

# After scanning all files fail if any broken links were found
if (( overall_errors > 0 )); then
  echo "Scan complete. Found $overall_errors broken links across all files."
  exit 1
else
  echo "Scan complete. All links are valid across all files."
  exit 0
fi
