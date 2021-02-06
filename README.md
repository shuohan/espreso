# Slice Selection Profile Estimation

#### Example

```bash
data_dir=/path/to/data
image=${data_dir}/image.nii.gz
output_dir=${data_dir}/output_dir

docker run --gpus device=0 --rm -t \
    -v $data_dir:$data_dir \
    --user $(id -u):$(id -g) \
    ssp train2d.py -i $image -o $outdir -l 19 \            
    -sw 0.3 -isz 1 -bs 16 -e 30000 -w 0 -bw 50      
```
