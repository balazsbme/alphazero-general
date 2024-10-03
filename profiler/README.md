# Profiler

## Usage

```bash
python3 profiler/profiler-sweep-params.py
```

## Plot running times

Plot running times for each combination based on the output logs generated by the parameter sweep.

```bash
cat profiler/monitor_20240923_233959.log | grep "Running combination " > profiler/starting-lines.txt
python3 profiler/plot-combination-running-times.py profiler/starting-lines.txt
```