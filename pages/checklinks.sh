#!/bin/bash
# Usage: ./checklinks.sh [FILES]
# Scans markdown files for broken links. Defaults to recursively finding "*.md" files if no FILES are provided.
# Only works for markdown links, and only absolute (http/https) links

# Exit on non-zero exit code for any calls.
set -e

# Directory or file pattern to scan (can be passed as a command-line argument).
# Defaults to all "*.md" under current directoru using `find`.
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 [FILES]"
  echo "Scans markdown files for broken links."
  echo "  FILES: Optional. Specify files or glob patterns to scan. Defaults to recursively finding all "*.md" files under the current directory."
  exit 0
fi
FILES=${1:-$(find . -type f -name "*.md")}

# grep -E regex to extract markdown links from markdown files - [text](link)
LINK_REGEX='\[[^]]+\]\((https?:\/\/[^[:space:]()]+(\([^[:space:]()]*\))?[^[:space:]()]*)\)'
# Regex to extract URL from markdown links
URL_REGEX='s/.*(\(http.*\)).*/\1/p'

# Function to check URL availability and log number of errors
check_url() {
  local url=$1
  echo "Checking URL: $url"
  
  # Try with GET request to get status code
  status_code=$(curl --silent --fail --max-time 15 -L -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" -o /dev/null -w "%{http_code}" "$url")
  
  # Handle returned status codes
  if [[ "$status_code" -eq 200 ]]; then
    echo "✅ $url is reachable (HTTP $status_code)."
    return 0
  elif [[ "$status_code" -eq 000 ]]; then
    echo "⚠️ Warning: $url returned HTTP $status_code (connection failed or timeout)."
    return 1
  elif [[ "$status_code" -eq 403 ]]; then
    # Check if this might be a Cloudflare challenge or other similar security measure
    response=$(curl --silent --max-time 10 -L -A "Mozilla/5.0" "$url" 2>/dev/null | grep -i "cloudflare\|just a moment\|enable javascript\|access denied" | head -1)
    if [[ -n "$response" ]]; then
      echo "⚠️ Warning: $url returned HTTP $status_code (likely protected by Cloudflare/bot protection)."
      echo "   This URL may be valid but requires a browser with JavaScript enabled."
    else
      # If we can't determine, treat as potentially valid but with access restrictions
      echo "⚠️ Warning: $url returned HTTP $status_code (may require special access or browser)."
      echo "   This URL might be valid but requires specific conditions to access."
    fi
  elif [[ "$status_code" -eq 429 ]]; then
    echo "❌ $url is not reachable (HTTP $status_code - Rate limited, try again later)."
    return 1
  elif [[ "$status_code" -eq 404 ]]; then
    echo "❌ $url is not reachable (HTTP $status_code - Not Found)."
    return 1
  elif [[ "$status_code" -ge 400 && "$status_code" -ne 403 ]]; then
    echo "❌ $url is not reachable (HTTP $status_code)."
    return 1
  elif [[ "$status_code" -ge 300 && "$status_code" -lt 400 ]]; then
    echo "⚠️ Warning: $url returned HTTP $status_code (redirect followed)."
  else
    echo "❌ $url returned unexpected HTTP $status_code."
    return 1
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
