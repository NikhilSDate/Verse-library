
set -m

cleanup() {
    echo "Received SIGINT, killing all processes..."
    # Kill all processes in the current process group
    kill -TERM -$$
    exit 0
}

trap cleanup SIGINT

for i in {1..40}; do
  timeout 2m python3 pump/verification.py -p 4 -o results/test
done
