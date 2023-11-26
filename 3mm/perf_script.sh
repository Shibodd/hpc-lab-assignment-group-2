
# Clean the old results file
RESULTS_FILE="results.txt"
OUTPUT_CSV="output.csv"
rm $OUTPUT_CSV
touch $OUTPUT_CSV
rm $RESULTS_FILE
touch $RESULTS_FILE


echo -e "\"sw_name\",\t\"secconds\",\t\"cache-misses\",\t\"cache-references\",\t\"context-switches\"\t\"insn per cycle\",\t\"CPUs utilized\"" > output.csv

# Benchmark
function bench() 
{
  rm $RESULTS_FILE
  touch $RESULTS_FILE
  module load perf

  perf stat -e context-switches,cache-references,cache-misses,instructions,cycles,task-clock ./$1 >> results.txt 2>&1
 
  #seconds=$(grep "seconds time elapsed" toparse.txt | awk '{print $1}')
  seconds=$(head -n 1 results.txt)
  cache_misses=$(grep "cache-misses" results.txt | awk '{print $1}')
  cache_references=$(grep "cache-references" results.txt | awk '{print $1}')
  context_switches=$(grep "context-switches" results.txt | awk '{print $1}')
  insn_per_cycle=$(grep "instructions" results.txt | awk '{print $4}')
  cpus_utilized=$(grep "CPUs utilized" results.txt | awk '{print $5}')

  echo -e "\"$1\",\t\"$seconds\",\t\"$cache_misses\",\t\"$cache_references\"\t\"$context_switches\"\t\"$insn_per_cycle\",\t\"$cpus_utilized\"" >> output.csv

  echo "CSV creato con successo!"  
}

echo "-------- PARALLELO -------- "
bench "3mm_parallel_matmuls_opt_acc"

echo "-------- SEQUENZIALE --------"
bench "3mm_ref_acc"
