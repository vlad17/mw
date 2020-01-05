# Multiplicative Weights

## Setup

Requirements: [Anaconda 3](https://www.anaconda.com/distribution/) in the path.

Make sure that `conda` is available in your path. Then run `conda activate`.

**To be done when you clone or move this repo**:
```
rm -rf .condaenv
conda env remove --name mw-env || true
(cat environment.yaml && echo "prefix: $CWD/.condaenv") > .condaenv.yaml
conda env create -f .condaenv.yaml
```

**Should be done once per session:**
```
conda activate mw-env
```

**To save new dependencies**:
```
# don't save prefix (computer specific)
# or debug-relevant stuff
conda env export --no-builds | grep -v "prefix: " > environment.yaml
```

**To update out-of-date local environment with new dependencies in the `environment.yaml` file**:
```
conda env update -f environment.yaml --prune
```

## Usage

All modules are expected to be run from the root directory of the repository

```
# use local ray cluster
REDIS_PORT=6379
ray start --num-cpus $(($(nproc) - 1)) --redis-port $REDIS_PORT --head
RAY_ADDRESS="$(python -c 'import ray;print(ray.services.get_node_ip_address())'):$REDIS_PORT"
function submit() {
  python scripts/online.py $@
}
function asubmit() {
  submit "$@"
}

# use aws
ray up config/ray-aws.yaml
REPO_ROOT="~/mw-repo" # root on ray head node
RAY_ADDRESS="localhost:6379"
function submit() {
  ray submit config/ray-aws.yaml scripts/online.py --args="$@"
}
function asubmit() {
  ray submit config/ray-aws.yaml --tmux scripts/online.py --args="$@"
}

# generate game matrix with a small warmup
for game in "--n 25 --m 40" "--n 10 --m 100" "--n 5 --m 200" ; do
  shortname="$(echo $game | tr -d '-' | tr -d ' ')"
  args=" --outfile $REPO_ROOT/data/${shortname} "
  args="$args $game --T 100 --eps 0.1 --ray_address $RAY_ADDRESS"
  submit "$args"
done

# submit jobs to ray for sim
for game in "--n 25 --m 40" "--n 10 --m 100" "--n 5 --m 200" ; do
  shortname="$(echo $game | tr -d '-' | tr -d ' ')"
  for eps in 0.1 0.01 0.001 ; do 
    fname="${shortname}eps${eps}"
    args="--outfile $REPO_ROOT/data/$fname"
    args="$args --T 100000000 --eps $eps"
    args="$args $game --ray_address $RAY_ADDRESS"
    args="$args --reuse_game_matrix $REPO_ROOT/data/${shortname}.npz"
    asubmit "$args"
  done
done

for game in "--n 25 --m 40" "--n 10 --m 100" "--n 5 --m 200" ; do
  shortname="$(echo $game | tr -d '-' | tr -d ' ')"
  args="--outfile $REPO_ROOT/data/${shortname}eps0.5decay0.5"
  args="$args --T 100000000 --eps 0.5 --decay 0.5"
  args="$args $game --ray_address $RAY_ADDRESS"
  args="$args --reuse_game_matrix $REPO_ROOT/data/${shortname}.npz"
  asubmit "$args"
done

ray rsync-down ./config/ray-aws.yaml "~/mw-repo/data" ./data

# plot into ./data/plot*.pdf
for game in "--n 25 --m 40" "--n 10 --m 100" "--n 5 --m 200" ; do
  shortname="$(echo $game | tr -d '-' | tr -d ' ')"
  flags="--outfile data/plot${shortname}.pdf"
  for eps in 0.1 0.01 0.001 ; do 
    flags="$flags --runs ./data/${shortname}eps${eps}"
  done
  flags="$flags --runs ./data/${shortname}eps0.5decay0.5"
  plotcmd="python scripts/optimality_gap.py $flags"
  echo $plotcmd
  $plotcmd
done
```
