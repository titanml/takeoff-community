#!/bin/bash
DIR="$(dirname "$(realpath "$0")")"
set -x
set -e

parallel=0
random_topk=0
output_file="output.json"
target="http://localhost:8000"
while getopts ":t:r:po:T:" opt; do
  case $opt in
    t) text="$OPTARG"
    ;;
    r) requests="$OPTARG"
    ;;
    p) parallel=1
    ;;
    o) output_file="$OPTARG"
    ;;
    T) target="$OPTARG"
    ;;
    \?) display_usage
    ;;
  esac
done

function display_usage() {
  echo "Usage: $0 -r <arg1> -t <arg2> -p -T <arg3>"
  echo "  -r: The number of requests to send"
  echo "  -t: The text body to send with each request"
  echo "  -p: Run the requests in parallel (optional)"
  echo "  -T: The target server to send requests to (optional, defaults to http://localhost:8000)"
  exit 1
}

function wait_for_server() {
  sleep 5
  while :
  do
    response=$(curl --write-out '%{http_code}' --silent --output /dev/null $target/docs)
    if [ "$response" -eq 200 ]; then
      break
    else
      echo "Server not ready, sleeping..."
      sleep 5
    fi
  done
}


if [[ -z $requests || -z $text ]]; then
  echo "Error: Both arguments are required."
  display_usage
fi

# Wait for the server to be ready before sending requests
wait_for_server

declare -a timings

function request() {
  timing=$(curl -s -o /dev/null -w "%{time_total}\n" -X POST -H "Content-Type: application/json" -d "{\"text\": \"$text\",  \"generate_max_length\": 128, \"sampling_topk\": 1, \"sampling_topp\": 1, \"sampling_temperature\": 1, \"repetition_penalty\": 1, \"no_repeat_ngram_size\": 0}" $target/generate)
  echo "$timing"
}

export -f request
export text
export requests
export target

start_time=$(date +%s%N) # Start time in nanoseconds

if [ "$parallel" -eq 1 ]; then
  # Run requests in parallel
  seq "$requests" | parallel -j$(nproc) --no-notice request > timings.txt
  timings=($(<timings.txt))
else
  # Run requests sequentially
  for ((i = 1; i <= $requests; i++))
  do
    echo "Sending request $i"
    timings+=("$(request)")
  done
fi

end_time=$(date +%s%N) # End time in nanoseconds
total_time=$(awk "BEGIN {print ($end_time - $start_time) / 1000000000}") # Calculate total time in seconds

min_time=$(printf '%s\n' "${timings[@]}" | sort -n | head -n1)
max_time=$(printf '%s\n' "${timings[@]}" | sort -n | tail -n1)

average_time=$(awk "BEGIN {print $total_time/$requests; exit}")

# Calculate the 95th and 99th percentile
sorted_timings=($(printf '%s\n' "${timings[@]}" | sort -n))
p95_index=$(awk "BEGIN {print int((0.95 * ${#timings[@]}) + 0.5) - 1}")
p99_index=$(awk "BEGIN {print int((0.99 * ${#timings[@]}) + 0.5) - 1}")

p95=${sorted_timings[$p95_index]}
p99=${sorted_timings[$p99_index]}

# Collect machine details
num_cores=$(nproc)
total_memory=$(free -m | awk '/^Mem:/{print $2}') # in MB

# Append to the JSON output
echo "{}" | jq --arg r "$requests" --arg min "$min_time" --arg max "$max_time" --arg avg "$average_time" --arg tot "$total_time" --arg p95 "$p95" --arg p99 "$p99" --arg cores "$num_cores" --arg mem "$total_memory" --arg gpu "$gpu_info" \
  '. + {requests: $r, min_time: $min, max_time: $max, average_time: $avg, total_time: $tot, p95: $p95, p99: $p99, num_cores: $cores, total_memory: $mem}' | tee $output_file


