# Baselines

Each baseline is a patch on its official repository at the commit below, with a `RUN.md` (setup,
data, commands) and the conda `environment.yml` used. `track_a/hf_to_colmap.py` and
`track_b/hf_to_wai.py` convert the Hugging Face release into the input of Track A and Track B. The
release has 93 scenes; the benchmark uses 91 (all except `2023/Day088_Corn` and `2023/Day088_Soy`).

| Baseline | Official repository | Commit |
|---|---|---|
| Mip-Splatting | https://github.com/autonomousvision/mip-splatting | `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9` |
| Scaffold-GS | https://github.com/city-super/Scaffold-GS | `59c833b56bbbf510f3f64d40f81721995caced66` |
| CityGaussian | https://github.com/Linketic/CityGaussian | `e84c7c8774dd11d3f4189be3488e1220afa20a86` |
| MapAnything (Track B) | https://github.com/facebookresearch/map-anything | `f065d13383ae84a7f80f0e0104a4d12afc7c7c82` |
