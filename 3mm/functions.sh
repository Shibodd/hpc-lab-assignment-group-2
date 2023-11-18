REFDUMP_PATH="reference_dump.txt"
DUMP_PATH="dump.txt"
VALIDATION_DATASET="SMALL"

# Runs the benchmark redirecting stderr
# ARG 1: Benchmark name
# ARG 2: Stderr redirection target
# OUTPUT time: The execution time
function single_run
{
  local EXEC=./"$1"_acc

  if ! test -f "$EXEC";
  then
    echo "== File $EXEC does not exist!" >&2
    exit 1;
  fi

  echo "$EXEC" >&2
  time=$("$EXEC" 2> $2)
}

# Deletes the executable of the benchmark
# ARG 1: Benchmark name
function single_clean
{
  make clean BENCHMARK=$1
}

# Builds the benchmark for validation (mini dataset and dump arrays)
# ARG 1: Benchmark name
function single_validation_build
{
  echo "== Building $1 for validation"
  single_clean $1
  make CFLAGS="-DPOLYBENCH_DUMP_ARRAYS -D$VALIDATION_DATASET$\_DATASET" BENCHMARK=$1
}

# Builds the benchmark for benchmarking.
# ARG 1: Benchmark name
# ARG 2: Additional CFLAGS
function single_bench_build
{
  echo "== Building $1 for benchmarking with \"$2\"" >&2
  single_clean $1
  make CFLAGS="-DPOLYBENCH_TIME $2" BENCHMARK=$1
}

# Cleans, builds and validates the benchmark
# ARG 1: Benchmark name
# OUTPUT time: Empty
function single_validation
{
  echo "== Validating $1" >&2
  single_validation_build $1
  single_run $1 $DUMP_PATH
  single_clean $1

  if ! cmp --silent "$DUMP_PATH" "$REFDUMP_PATH";
  then
    echo "== Bad output! Check $DUMP_PATH" >&2
    exit 1;
  fi
}

# Cleans, builds, runs the benchmark and dumps stderr to REFDUMP_PATH
# ARG 1: Benchmark name
# OUTPUT time: The execution time
function single_make_reference_dump
{
  echo "== Generating reference dump for $1" >&2
  single_validation_build $1
  single_run $1 $REFDUMP_PATH
  single_clean $1
}

# Runs the benchmark
# ARG 1: Benchmark name
# OUTPUT time: The execution time
function single_bench
{
  echo "== Benchmarking $1"
  single_run $1 /dev/null
  single_clean $1
}