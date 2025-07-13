while true; do
  timeout 3m python3 pump/verification.py -p 4 -o results/test
done
