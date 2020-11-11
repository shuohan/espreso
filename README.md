# Point Spread Function Estimation

1. Optimize the network to output a Gaussian kernel


Build docs

```bash
docker run --rm -v $(realpath $PWD):$(realpath $PWD) \
    --user $(id -u):$(id -g) -w $(realpath $PWD) -t \
    psf-est sphinx-build docs/source docs/build
```
