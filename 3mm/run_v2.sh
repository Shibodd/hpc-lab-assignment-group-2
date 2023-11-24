source functions.sh

# Create the reference dump
single_make_reference_dump "3mm_parallel_matmuls_opt"
single_make_reference_dump "3mm_ref"


# Clean the old results file
RESULTS_FILE="results.txt"
OUTPUT_CSV="output.csv"
rm $OUTPUT_CSV
touch $OUTPUT_CSV
rm $RESULTS_FILE
touch $RESULTS_FILE


echo -e "\"sw_name\",\t\"cache-misses\",\t\"cache-references\",\t\"context-switches\"\t\"insn per cycle\",\t\"CPUs utilized\"\n" > output.csv

# Benchmark
function bench()
{
  rm $RESULTS_FILE
  touch $RESULTS_FILE
  # Validate the source first
  single_validation $1

  # Then benchmark with different configurations
  if [ $1 = "3mm_parallel_matmuls_opt" ]; then
    local flags="-DMEDIUM_DATASET -fopenmp";
  else
    local flags="-DMEDIUM_DARASET";
  fi
  flags="-DMEDIUM_DATASET -fopenmp";
  single_bench_build $1 "$flags"
  #single_bench_build $2 "$flags"
  #single_bench $1
  #echo $1,$flags,$time | tee -a $RESULTS_FILE
  module load perf
  
  #perf stat ./$1_acc >toparse.txt 2>&1

  perf stat -e context-switches,cache-references,cache-misses,instructions,cycles,task-clock ./$1_acc >> results.txt 2>&1
 
  #seconds=$(grep "seconds time elapsed" toparse.txt | awk '{print $1}')
  cache_misses=$(grep "cache-misses" results.txt | awk '{print $1}')
  cache_references=$(grep "cache-references" results.txt | awk '{print $1}')
  context_switches=$(grep "context-switches" results.txt | awk '{print $1}')
  insn_per_cycle=$(grep "instructions" results.txt | awk '{print $4}')
  cpus_utilized=$(grep "CPUs utilized" results.txt | awk '{print $5}')

  echo -e "\"$1\",\t\"$cache_misses\",\t\"$cache_references\"\t\"$context_switches\"\t\"$insn_per_cycle\",\t\"$cpus_utilized\"\n" >> output.csv

  echo "CSV creato con successo!"  
}

echo "-------- PARALLELO -------- "
bench "3mm_parallel_matmuls_opt"

echo "-------- SEQUENZIALE --------"
bench "3mm_ref"
