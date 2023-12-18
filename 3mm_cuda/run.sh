set -e

if [ -z "$1" ]
then
  echo "Usage: $0 <dataset_size> [HOST]"
  echo "Where dataset_size is one of MINI, SMALL, MEDIUM, STANDARD, LARGE, EXTRALARGE"
  echo "Use uppercase HOST as second argument to specify that you want the CPU code to be profiled, too"
  exit
fi

make clean all profile EXERCISE=3mm_ref_g.cu EXT_CXXFLAGS="-D$1""_DATASET -DPOLYBENCH_TIME -DPOLYBENCH_GFLOPS -DM_$2"