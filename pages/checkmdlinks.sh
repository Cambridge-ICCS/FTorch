#!/bin/bash
# Usage: ./checkmdlinks.sh [FILES]
# Scans markdown files for broken links. Defaults to recursively finding "*.md"
# files under the current directory if no FILES are provided.
#
# Only checks markdown links (i.e. [text](url)) and only absolute http/https URLs;
# relative links and bare plaintext URLs are ignored.
#
# Exit code is 0 if all links are reachable, 1 otherwise. In GitHub Actions
# (`$GITHUB_ACTIONS == "true"`) a single aggregated `::notice` workflow command
# is emitted listing any broken links; the workflow step that invokes this
# script is marked `continue-on-error: true` so a flaky external site cannot
# fail the build.

# Exit on non-zero exit code for any calls.
set -e

# Directory or file pattern to scan (can be passed as a command-line argument).
# Defaults to all "*.md" under current directory using `find`.
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 [FILES]"
  echo "Scans markdown files for broken links."
  echo "  FILES: Optional. Specify files or glob patterns to scan."
  echo "         Defaults to recursively finding all \"*.md\" files under the current directory."
  exit 0
fi
FILES=${1:-$(find . -type f -name "*.md")}

# grep -E regex to extract markdown links from markdown files - [text](link)
LINK_REGEX='\[[^]]+\]\((https?://[^[:space:]()]+(\([^[:space:]()]*\))?[^[:space:]()]*)\)'
# Regex to extract URL from markdown links
URL_REGEX='s/.*(\(http.*\)).*/\1/p'

# Accumulate failing URLs as "url file http_status" lines for the summary.
FAILED_LINKS=()

# Function to check URL availability and log number of errors
check_url() {
  local url=$1
  local file=$2
  echo "Checking URL: $url"

  # Try with GET request to get status code
  status_code=$(curl --silent --fail --max-time 15 -L -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" -o /dev/null -w "%{http_code}" "$url")

  # Handle returned status codes
  if [[ "$status_code" -eq 200 ]]; then
    echo "✅ $url is reachable (HTTP $status_code)."
    return 0
  elif [[ "$status_code" -eq 000 ]]; then
    echo "⚠️ Warning: $url returned HTTP $status_code (connection failed or timeout)."
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
  elif [[ "$status_code" -eq 404 ]]; then
    echo "❌ $url is not reachable (HTTP $status_code - Not Found)."
  elif [[ "$status_code" -ge 400 && "$status_code" -ne 403 ]]; then
    echo "❌ $url is not reachable (HTTP $status_code)."
  elif [[ "$status_code" -ge 300 && "$status_code" -lt 400 ]]; then
    echo "⚠️ Warning: $url returned HTTP $status_code (redirect followed)."
    return 0
  else
    echo "❌ $url returned unexpected HTTP $status_code."
  fi

  FAILED_LINKS+=("$url|$file|$status_code")
  return 1
}

# Extract and check URLs
for file in $FILES; do
  echo "=============================================================================="
  echo "Scanning file: $file"
  echo "=============================================================================="
  file_errors=0
  if ! grep -Eo "$LINK_REGEX" "$file" > /dev/null; then
    echo "No matches found in $file"
  fi
  while IFS= read -r url; do
    if ! check_url "$url" "$file"; then
      file_errors=$((file_errors + 1))
    fi
  done < <(grep -Eo "$LINK_REGEX" "$file" | sed -n "$URL_REGEX")
  if (( file_errors > 0 )); then
    echo "Found $file_errors broken links in $file."
  else
    echo "All links in $file are valid."
  fi
  echo -e "\n"
done

# Summary
overall_errors=${#FAILED_LINKS[@]}
if (( overall_errors == 0 )); then
  echo "Scan complete. All links are valid across all files."
  exit 0
fi

echo "Scan complete. Found $overall_errors broken links across all files:"
for entry in "${FAILED_LINKS[@]}"; do
  IFS='|' read -r url file status <<< "$entry"
  echo "  - $url (HTTP $status) in $file"
done

# In GitHub Actions emit a single aggregated ::notice so the failure is visible
# in the workflow summary without breaking the job (the step is set
# continue-on-error: true). Escape newline as %0A per workflow-command syntax.
if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
  notice_msg="Possible broken links in documentation (this check is non-blocking - please verify manually):"
  for entry in "${FAILED_LINKS[@]}"; do
    IFS='|' read -r url file status <<< "$entry"
    notice_msg+="%0A  - $url (HTTP $status) in $file"
  done
  echo "::notice title=Possibly broken documentation links::$notice_msg"
fi

exit 1
