
# set -m

# cleanup() {
#     echo "Received SIGINT, killing all processes..."
#     # Kill all processes in the current process group
#     kill -TERM -$$
#     exit 0
# }

# trap cleanup SIGINT

for i in {1..2}; do
  timeout --signal=SIGINT 1m python3 pump/verification.py -p 1 -o results/test_local -n $1 -i $2
done
