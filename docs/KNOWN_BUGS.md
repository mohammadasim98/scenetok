# Bug Report


## 1. Incorrect scaling factor

In the following experiments, we incorrectly scale the focal point by a factor defined in `src/misc/dl3dv_utils.py; line:45`

```bash
scenetok_va-vdc_shift4_dl3dv_finetuned
scenetok_va-vdc_shift8_dl3dv_finetuned
```

The intrinsics are already normalized by the stored height and width therefore does not require additional scaling. We dont have this bug in other experiments.


## 2. Incorrect scaling factor for context

Similar to [above](#1-incorrect-scaling-factor/), we incorrectly scale the focal point of the context views only since the experiments used the same precomputed latents for VA-VAE as in the previous issue

```bash
scenetok_va-wan_shift4_dl3dv_finetuned
scenetok_va-wan_shift8_dl3dv_finetuned
```


## 3. Incorrect `temporal_downsample` factor passed to camera embedder

In `src/model/embedding/lvsm_embed.py`, we provide with the parameter `temporal_downsample` for the reshapping layer as shown in the following,

```python
nn.Sequential(
    Rearrange(
        "b ... (t c) (hh ph) (ww pw) -> b ... (hh ww) (t ph pw c)",
        ph=cfg.patch_size,
        pw=cfg.patch_size,
        t=temporal_downsample
    ),
    nn.Linear(
        cfg.in_channels * (cfg.patch_size**2),
        embed_dim,
        bias=False,
    ),
    # nn.RMSNorm(embed_dim),
)
```

Normally the correct value should be `temporal_downsample=4` for both VideoDCAE and WanVAE, but the model we trained for WanVAE has `temporal_downsample=1`. The following configs are affected by this bug, (same as in [2](#2-incorrect-scaling-factor-for-context))

```bash
scenetok_va-wan_shift4_dl3dv_finetuned
scenetok_va-wan_shift8_dl3dv_finetuned
```



> [!INFO] 
> In all future experiments, we will address these bugs, but for the trained model so far, we added certain parameters to allow these intentionally for inference. Make sure you dont enable them accidently when training your own model from scratch.

```yaml
dataset.scale_focal_by_256: true # for Bug: 1
dataset.scale_context_focal_by_256: true # for Bug: 2
model.force_incorrect: true # for Bug: 3
```
