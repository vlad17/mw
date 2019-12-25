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
conda env export --no-builds | grep -v "prefix: " > environment.yaml
```

**To update out-of-date local environment with new dependencies in the `environment.yaml` file**:
```
conda env update -f environment.yaml --prune
```

## Usage

All modules are expected to be run from the root directory of the repository

```
# re-use the same ray cluster between invocations
REDIS_PORT=6379
ray start --num-cpus $(($(nproc) - 1)) --redis-port $REDIS_PORT --head
RAY_ADDRESS="$(python -c 'import ray;print(ray.services.get_node_ip_address())'):$REDIS_PORT"

# generate game matrix with a small warmup
for game in "--n 25 --m 40" "--n 10 --m 100" "--n 5 --m 200" ; do
  shortname="$(echo $game | tr -d '-' | tr -d ' ')"
  python scripts/online.py --outfile "./data/${shortname}" \
    $game --T 100 --eps 0.1 --ray_address $RAY_ADDRESS
done

# submit jobs to ray for sim
for game in "--n 25 --m 40" "--n 10 --m 100" "--n 5 --m 200" ; do
  shortname="$(echo $game | tr -d '-' | tr -d ' ')"
  for eps in 0.1 0.01 0.001 ; do 
    python scripts/online.py \
      --outfile "./data/${shortname}eps${eps}" \
      --T 100000000 --eps $eps \
      $game \
      --ray_address $RAY_ADDRESS \
      --reuse_game_matrix "./data/${shortname}.npz" &
  done
done ; \
wait

# plot into ./data/plot*.pdf
for game in "--n 25 --m 40" "--n 10 --m 100" "--n 5 --m 200" ; do
  shortname="$(echo $game | tr -d '-' | tr -d ' ')"
  flags="--oufile data/plot${shortname}.pdf"
  for eps in 0.1 0.01 0.001 ; do 
    flags="$flags --runs \"./data/${shortname}eps${eps}\""
  done
  plotcmd="python scripts/optimality_gapy.py $flags"
  echo $plotcmd
  $plotcmd
done
```
