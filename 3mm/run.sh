source functions.sh

# Create the reference dump
single_make_reference_dump "3mm_ref"

# Clean the old results file
RESULTS_FILE="results.txt"
rm $RESULTS_FILE
touch $RESULTS_FILE

# Benchmark
function bench()
{
  # Validate the source first
  single_validation $1

  # Then benchmark with different configurations
  local flags="";
  single_bench_build $1 "$flags"
  single_bench $1
  echo $1,$flags,$time | tee -a $RESULTS_FILE
}

bench "3mm_ref"