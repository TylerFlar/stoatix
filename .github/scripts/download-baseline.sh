#!/usr/bin/env bash
# Download the latest successful baseline artifact from main branch.
#
# Requires:
#   - gh CLI authenticated (GH_TOKEN env var or gh auth login)
#   - jq installed
#   - GITHUB_REPOSITORY set (automatic in GitHub Actions)
#
# Outputs:
#   - baseline.zip in current directory (caller should unzip)

set -euo pipefail

# Configuration (can be overridden via environment)
BASELINE_WORKFLOW_FILE="${BASELINE_WORKFLOW_FILE:-stoatix-baseline.yml}"
BASELINE_ARTIFACT_NAME="${BASELINE_ARTIFACT_NAME:-stoatix-baseline}"
BASELINE_BRANCH="${BASELINE_BRANCH:-main}"

# Ensure GITHUB_REPOSITORY is set
if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
    echo "::error::GITHUB_REPOSITORY environment variable is not set"
    exit 1
fi

echo "Searching for latest successful run of '$BASELINE_WORKFLOW_FILE' on '$BASELINE_BRANCH'..."

# Get the latest successful workflow run on the target branch
run_response=$(gh api \
    "repos/$GITHUB_REPOSITORY/actions/workflows/$BASELINE_WORKFLOW_FILE/runs?branch=$BASELINE_BRANCH&status=success&per_page=1" \
    --jq '.')

run_count=$(echo "$run_response" | jq '.total_count')
if [[ "$run_count" -eq 0 ]]; then
    echo "::error::No successful runs found for workflow '$BASELINE_WORKFLOW_FILE' on branch '$BASELINE_BRANCH'"
    echo "::error::Make sure the baseline workflow has run successfully at least once on '$BASELINE_BRANCH'"
    exit 1
fi

run_id=$(echo "$run_response" | jq -r '.workflow_runs[0].id')
run_url=$(echo "$run_response" | jq -r '.workflow_runs[0].html_url')
run_date=$(echo "$run_response" | jq -r '.workflow_runs[0].created_at')

echo "Found run #$run_id from $run_date"
echo "  URL: $run_url"

# List artifacts for that run
echo "Searching for artifact '$BASELINE_ARTIFACT_NAME'..."

artifacts_response=$(gh api \
    "repos/$GITHUB_REPOSITORY/actions/runs/$run_id/artifacts" \
    --jq '.')

artifact_id=$(echo "$artifacts_response" | jq -r \
    --arg name "$BASELINE_ARTIFACT_NAME" \
    '.artifacts[] | select(.name == $name) | .id')

if [[ -z "$artifact_id" ]]; then
    echo "::error::Artifact '$BASELINE_ARTIFACT_NAME' not found in run #$run_id"
    echo "Available artifacts:"
    echo "$artifacts_response" | jq -r '.artifacts[].name' | sed 's/^/  - /'
    exit 1
fi

artifact_size=$(echo "$artifacts_response" | jq -r \
    --arg name "$BASELINE_ARTIFACT_NAME" \
    '.artifacts[] | select(.name == $name) | .size_in_bytes')

echo "Found artifact #$artifact_id ($(numfmt --to=iec-i --suffix=B "$artifact_size" 2>/dev/null || echo "${artifact_size} bytes"))"

# Download the artifact
echo "Downloading artifact..."

gh api \
    "repos/$GITHUB_REPOSITORY/actions/artifacts/$artifact_id/zip" \
    > baseline.zip

if [[ ! -f baseline.zip ]]; then
    echo "::error::Failed to download artifact"
    exit 1
fi

zip_size=$(stat --printf="%s" baseline.zip 2>/dev/null || stat -f%z baseline.zip 2>/dev/null || echo "unknown")
echo "Downloaded baseline.zip ($zip_size bytes)"

echo ""
echo "✅ Successfully downloaded baseline artifact"
echo "   Run: #$run_id ($run_date)"
echo "   Artifact: $BASELINE_ARTIFACT_NAME"
echo ""
echo "Next: unzip baseline.zip -d baseline_out/"
