# Dataset Format

## Training
In our experiments, we use `latent` datasets where we precompute and store the latents for both *context* and *target* views on disk (ty) for faster training and high throughput. We additionally also provide with a standard RGB version, 

> [!TIP]
> we recommend using our latent dataloaders when training `SceneTok` and `SceneGen`.

## Inference
Inference does not need to use the latent dataset and instead use RGB dataset and can use the original dataset format.
> For RE10K, we follow [pixelSplat](https://github.com/dcharatan/pixelsplat/)

> For DL3DV, use our custom dataloader class in `src.dataset.dl3dv.DL3DVDataset`. Data is available [here](https://github.com/DL3DV-10K/Dataset)

### Using Precomputed Latents


#### Format
We use the following format for each latent dataset (VA-VAE, VideoDCAE, WanVae, etc.)

```bash
{root}/
    {stage}/
        {downsample_factor}/
            {scene_name}["_flipped"]*.npz
```


- `stage` can be either train or test

- `downsample_factor` is an integer factor used to downsample the full video sequence **before** computing latents. This is useful when using `videodcae` as they compress the RGB frames temporally as well. 
    - In our experiments we set  `downsample_factor=1` for both context and targets

- `scene_name` is the unique id assigned to individual scenes. We use one file per scene and if `flip=true` then we include an additional suffix `_flipped` in the file name.
- We store all files in `.npz` format with minimal compression and with `allow_pickle=False`.


#### Preprocessing raw data to compute latents

The main entry point for the preprocessing script is `src.scripts.preprocess_dataset` with the config is `config/scripts/preprocess_config.yaml` 
```bash
python -m src.scripts.preprocess_dataset \
    dataset=${specific_configs} \ 
    stage=${stage} \
    index=${index} \
    size=${size} \
    output_dir=${output_dir} \
    flip=${flip}
```
- `specific_config` are configs defined in the directory `config/scripts/dataset` which corresponds to different autoencoders for RealEstate10K and DL3DV.
- `stage` is defined as either `test` or `train`
- `index` is a useful parameter for parallelizing the preprocessing steps across multiple GPUs (e.g., on SLURM). If you decide to run a single preprocessing script, then set `index=0`. 
- `size` defines the number of `.torch` chunks to process in case of **RealEstate10K** where each chunk has multiple scenes (c.f. [pixelSplat](https://github.com/dcharatan/pixelsplat/)) whereas in case of **DL3DV**, it refers to the *number of individual scene*. 
- `output_dir` is the output directory to which you want to save the computed latents following the same structure as [above](#format)
- `flip` either `true` or `false` which enables flipped version of the video sequence. The flip operation is performed on each **image** in the sequence and the **extrinsics** are also flipped 

### Some Examples: 

1. For RealEstate10K:
    ```bash
    # <============= VA-VAE =============>
    # Test
    python -m src.scripts.preprocess_dataset dataset=va_re10k \
        dataset.data_root=${root} \
        stage=test \
        index=0 \
        size=600 \
        output_dir="./data/preprocessed_data/va_re10k" \ 
        flip=false
    
    # Train
    python -m src.scripts.preprocess_dataset dataset=va_re10k \
        dataset.data_root=${root} \
        stage=train \
        index=0 \
        size=5000 \
        output_dir="./data/preprocessed_data/va_re10k" \ 
        flip=false
    
    # <============= VideoDCAE =============>
    # Test
    python -m src.scripts.preprocess_dataset dataset=videodc_re10k \
        dataset.data_root=${root} \
        stage=test \
        index=0 \
        size=600 \
        output_dir="./data/preprocessed_data/videodc_re10k" \
        flip=false

    # Train
    python -m src.scripts.preprocess_dataset dataset=videodc_re10k \
        stage=train \
        index=0 \
        size=5000 \
        output_dir="./data/preprocessed_data/videodc_re10k" \
        flip=false
    ```

1. For DL3DV:
    ```bash

    # NOTE: Select a subset from ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "11K"] when converting the training data 

    # <============= VA-VAE =============>
    # Test
    python -m src.scripts.preprocess_dataset dataset=va_dl3dv \
        dataset.data_root=${root} \
        stage=test \
        index=0 \
        size=140 \
        output_dir="./data/preprocessed_data/va_dl3dv" \
        flip=false # Test
    
    # Train
    python -m src.scripts.preprocess_dataset dataset=va_dl3dv \
        dataset.data_root=${root} \
        dataset.subset=${subset} \
        stage=train \
        index=0 \
        size=1000 \
        output_dir="./data/preprocessed_data/va_dl3dv" \
        flip=false # Train

    # <============= VideoDCAE =============>
    # Test
    python -m src.scripts.preprocess_dataset dataset=videodc_dl3dv \
        dataset.data_root=${root} \
        stage=test \
        index=0 \
        size=140 \
        output_dir="./data/preprocessed_data/videodc_dl3dv" \
        flip=false # Test
    
    # Train
    python -m src.scripts.preprocess_dataset dataset=videodc_dl3dv \
        dataset.data_root=${root} \
        dataset.subset=${subset} \
        stage=train \
        index=0 \
        size=1000 \
        output_dir="./data/preprocessed_data/videodc_dl3dv" \
        flip=false # Train
    ```

> [!INFO] 
> In our experiments (for training only), we compute latent for both `flip=false` and `flip=true`  

### SLURM Scripts for preprocessing

We provide easy to use script to preprocess the complete dataset. In `scripts/run_process_*.sh`, set the correct root directory, slurm configurations and the environment settings.

First set the `PROJECT_ROOT` as environment variable:
  ```bash
  export PROJECT_ROOT="<your-project-directory>"
  ```

```bash
# DL3DV
bash convert_dl3dv_vavae.sh ${output_dir}
bash convert_dl3dv_videodc.sh ${output_dir}

# RE10K
bash convert_re10k_vavae.sh ${output_dir}
bash convert_re10k_videodc.sh ${output_dir}
```

### Post-Processing
Make sure to generate a map dictionary with:

```bash
python -m src.scripts.create_map_dict --root <your-latent-dataset-root>
```
This will create the dictionary files that will be used in `src.dataset.dataset_latent` via the config parameter `dataset.map_dict`.
The dictionary takes the following format:

```
{   
    "train": {
        ${downsample_factor}: {
            scene_id: [file1.npz, file2.npz]
        }
    }
    "test": {
        ${downsample_factor}: {
            scene_id: [file1.npz, file2.npz]
        }
    }

}
```
where `file1.npz` and `file2.npz` are separate augmentation of the same scene. In our case, we only limit to horizontal flip operation.

