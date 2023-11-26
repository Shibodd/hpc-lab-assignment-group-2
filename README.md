# hpc-lab-assignment-group-2

Go to ./3mm

************ COMPILATION AND EXECUTION ************

run this command just once when you open the terminal
  source functions.sh

Compile
  single_bench_build "3mm_parallel_matmuls_opt" "-fopenmp -DSTANDARD_DATASET"

Execute
  ./3mm_parallel_matmuls_opt_acc


No need to use make

_____________________________________________________________________________

You can choose one of these names after the command single_bench_build.
And the you can execute them

3mm_ref                      # sequential code
3mm_parallel_matmuls         # parallel code
3mm_parallel_matmuls_opt     # parallel and optimized code


************ RESULT SCRIPT ************
run 
  ./perf_script.sh 
to calculate performances for the following executables
  3mm_parallel_matmuls_opt_acc
  3mm_ref_acc

a .csv file called "output.csv" will be created, reporting the datas.
