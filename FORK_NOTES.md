# Fork notes

Every intentional divergence from upstream `Tencent/Hunyuan3D-2`, with the symptom that motivated it.
Keep this file updated when you change tracked upstream files — it is the diff's changelog, and the basis for
any PR sent back upstream.

**Baseline:** upstream commit `f8db630` ("Update LICENSE").
**Verified against:** Python 3.10, torch 2.14.0+cu126, diffusers 0.40.0, transformers 5.17.0, Quadro RTX 6000 (24 GB).

Upstream was written against roughly diffusers 0.32 / transformers 4.48. Most breakage below comes from that gap,
so expect more of it when you bump versions again.

| # | Type | Files | One-liner |
|---|------|-------|-----------|
| 1 | Bug fix | `hy3dgen/texgen/utils/multiview_utils.py` | Texture generation refused to load on diffusers >= 0.34 |
| 2 | Bug fix | `gradio_app.py` | Textured OBJ export downloaded without its texture |
| 3 | Feature | `gradio_app.py`, `api_server.py` | `--texgen_subfolder` / `--tex_subfolder` to select the paint model |
| 4 | Bug fix | `hy3dgen/texgen/hunyuanpaint/pipeline.py` | Full paint model crashed under model cpu offload |
| 5 | New tool | `batch_gen.py` | Unattended folder-of-images to textured GLBs |
| 6 | Feature | `gradio_app.py` | Target face number is settable for textured generation |

---

## 1. Texture generation fails to load — `trust_remote_code`

**Symptom.** The Gradio app started but showed *"Texture Generation: Unavailable"* and logged:

```
Failed to load texture generator.
Please try to install requirements by following README.md
```

The message is misleading — nothing was missing from the install. `gradio_app.py` wraps the paint pipeline load in a
broad `except Exception` that prints this regardless of the real cause. Calling
`Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')` directly shows the truth:

```
ValueError: The directory .../hy3dgen/texgen/hunyuanpaint contains custom code in pipeline.py which must be
executed to correctly load the model. ... Pass `trust_remote_code=True` to allow loading remote code modules.
  -> RuntimeError: Something wrong while loading /home/.../models--tencent--Hunyuan3D-2/snapshots/...
```

**Cause.** `Multiview_Diffusion_Net.__init__` loads the paint model with
`DiffusionPipeline.from_pretrained(..., custom_pipeline=<local path>)`. Diffusers >= 0.34 refuses to execute any
`custom_pipeline` without an explicit `trust_remote_code=True`, including one on local disk.

**Fix.** Pass `trust_remote_code=True`. The "remote code" is this repo's own
`hy3dgen/texgen/hunyuanpaint/pipeline.py`, checked into the tree — not third-party code fetched from the Hub.

```python
pipeline = DiffusionPipeline.from_pretrained(
    multiview_ckpt_path,
    custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16,
    trust_remote_code=True)
```

This single change also fixes `api_server.py --enable_tex` and therefore `blender_addon.py`, which share the pipeline.

**Worth upstreaming?** Yes. Low risk, no behaviour change on old diffusers.

**Possible follow-up:** `gradio_app.py`'s `except Exception` should re-raise or print the traceback instead of
guessing "install your requirements" — that wrong guess is what made this look like a broken install.

## 2. Textured OBJ export loses its texture

**Symptom.** Export tab → `export_texture` checked → file type `obj` → the downloaded `.obj` imports into Blender
untextured.

**Cause.** trimesh writes an OBJ export as three files in the save folder (`textured_mesh.obj`, `material.mtl`,
`material_0.png`), but `on_export_click` returns only the single `.obj` path to the download button. The `.mtl` and
the PNG stay on the server. GLB is unaffected because it embeds the texture in one file.

**Fix.** New `pack_sidecar_files()` helper in `gradio_app.py`, applied to the textured-export branch: if the save
folder holds more than the mesh file, zip the folder and hand back the archive instead. Self-contained formats
(glb) are returned unchanged, so nothing about the GLB path changes. The archive is built in a *separate* folder —
`shutil.make_archive` writing into the directory it is zipping can capture the growing archive inside itself.

Result: `obj` → `textured_mesh.zip` containing `textured_mesh.obj`, `material.mtl`, `material_0.png`.

**Worth upstreaming?** Yes, though upstream may prefer a different UX (e.g. offering the files separately).

**Not addressed:** `ply` and `stl` are offered in `SUPPORTED_FORMATS` but neither carries a UV texture map, so
"export textured" for those formats is inherently lossy. Consider hiding the texture checkbox for them.

## 3. Selectable paint checkpoint

**Symptom.** Both apps hardcoded the paint model to the `from_pretrained` default,
`hunyuan3d-paint-v2-0-turbo` (step-distilled). There was no way to run the full `hunyuan3d-paint-v2-0` for higher
texture quality without editing source.

**Change.** New flags, defaulting to the previous behaviour so existing commands are unaffected:

```bash
python3 gradio_app.py ... --texgen_subfolder hunyuan3d-paint-v2-0   # Gradio
python api_server.py --enable_tex --tex_subfolder hunyuan3d-paint-v2-0   # REST / Blender addon
```

Measured on the same mesh (`assets/1.glb` + `assets/demo.png`): turbo **31.7 s**, full **38.3 s** — only ~20 %
slower per asset, so the full model is usually worth it for final assets. The full checkpoint is a separate ~5 GB
download on first use, and the two live in different HF subfolders, so switching flags mid-session re-downloads
nothing but does reload the pipeline.

Choices are restricted to the two keys present in `Hunyuan3DTexGenConfig.pipe_dict`; anything else raises `KeyError`
deep inside the config, so `argparse` rejects it up front instead.

Also in `gradio_app.py`: the header now prints the live paint checkpoint (it previously always said "Hunyuan3D-2"
regardless of what was loaded), and the per-generation `stats['model']['texgen']` embedded in the exported mesh
metadata records `path/subfolder` instead of just `path` — so a generated asset says which model made it.

## 4. Full paint model + `--low_vram_mode` crashes on a device mismatch

**Symptom.** With `hunyuan3d-paint-v2-0` (not turbo) *and* model cpu offload enabled, every texture run dies:

```
RuntimeError: Expected all tensors to be on the same device, but got tensors is on cuda:0,
different from other tensors on cpu (when checking argument in method wrapper_CUDA_cat)
  at hunyuanpaint/pipeline.py, denoise(): prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
```

**Cause.** Two upstream details combine:

1. That `torch.cat` sits behind `if (self.do_classifier_free_guidance) and (not self.is_turbo)`, so **only the
   non-distilled model reaches it**. The turbo model skips CFG entirely, which is why turbo never hits this.
2. `denoise()` calls `encode_prompt(..., self.do_classifier_free_guidance if self.is_turbo else False, ...)`.
   For the non-turbo model that argument is hardcoded `False`, so `encode_prompt` relocates `prompt_embeds` to the
   execution device but leaves `negative_prompt_embeds` where it was.

Without offload both tensors happen to be on the GPU already (they derive from `self.unet.learned_text_clip_gen`),
so the bug is invisible. `enable_model_cpu_offload()` parks the unet on the CPU until its forward runs — the
negatives stay on the CPU, `prompt_embeds` is moved to CUDA, and the `cat` fails.

**Fix.** Align the negatives with `prompt_embeds` immediately before the concatenation. Three lines, no behaviour
change when the tensors were already co-located.

**Worth upstreaming?** Yes — this makes the quality paint model usable on <24 GB cards, which is the whole point of
`--low_vram_mode`.

## 5. `batch_gen.py` — unattended batch generation

New file, no upstream equivalent. Turns a folder of images into textured GLBs with the models loaded once:

```bash
python3 batch_gen.py --input_dir images/ --output_dir out/
```

Per asset: background removal -> shape -> floater/degenerate cleanup -> decimate to `--target_face_num`
(default 10000) -> texture -> `out/<name>.glb`. Decimation runs **before** painting, mirroring `generation_all`
in `gradio_app.py`, so the UV unwrap and bake work on the lighter mesh.

Defaults to the full `hunyuan3d-paint-v2-0`; `--no_texture` skips loading the paint model altogether.
`--texture_size` rebuilds the `MeshRender` (its resolution is fixed in the pipeline constructor, so overriding the
config alone would not take effect).

Unattended behaviour: a failing asset writes its traceback to `out/logs/<name>.log` and the run continues;
`manifest.csv` is flushed after every asset; existing outputs are skipped so an interrupted run resumes by
re-running the same command; exit status is 1 if anything failed.

## 6. Target face number for textured generation

**Symptom.** *Simplify Mesh* in the Export tab is greyed out after **Gen Textured Shape**, so every textured mesh
the UI produces is stuck at 40000 faces with no way to ask for fewer.

**Why upstream disables it.** This one is correct behaviour, not a bug. `reduce_face()` applies pymeshlab's
`meshing_decimation_quadric_edge_collapse`, not the `_with_texture` variant, so it does not carry UV coordinates
through. Decimating an already-baked mesh rebuilds the geometry and invalidates the UV layout the texture was baked
against — you would get a lower-poly mesh with a scrambled texture. `on_export_click` therefore ignores
`reduce_face`/`target_face_num` entirely in its `export_texture` branch, and the callback at the end of
`btn_all.click` sets the checkbox to `interactive=False`.

**Change.** Decimation has to happen *before* texturing, which `generation_all` already does — it just called
`face_reduce_worker(mesh)` with `FaceReducer`'s hardcoded 40000 default. A **Target Face Number** slider in
Advanced Options now feeds that call, so the setting lands at the only point in the pipeline where it is safe.
Default stays 40000, so shipped behaviour is unchanged. The value is recorded in `stats['params']` and travels in
the exported mesh metadata.

The slider carries an `info` string explaining why the Export tab cannot do this, so the greyed-out checkbox stops
looking like a fault.

Verified through the Gradio API: requesting 3000 and 40000 produced meshes of exactly 3000 and 40000 faces, both
with intact 2048x2048 textures.

**Worth upstreaming?** Yes, and it pairs naturally with a tooltip on the disabled Simplify Mesh checkbox.

---

## Known upstream issues left alone

Not fixed, because fixing them is a behaviour change rather than a repair. Noted so they are not rediscovered:

- **The compiled `mesh_processor` extension is dead code.**
  `differentiable_renderer/mesh_render.py` does `from .mesh_processor import meshVerticeInpaint`. The leading dot
  makes that resolve to the in-tree pure-Python `differentiable_renderer/mesh_processor.py`, so the C++ extension
  built and installed by `differentiable_renderer/setup.py` is shadowed and never used. Texture inpainting runs the
  slow Python path. Dropping the dot would use the fast one, but needs benchmarking and a fallback for users who
  skipped the build step.
- **`MeshSimplifier` cannot work.** `hy3dgen/shapegen/postprocessors.py` shells out to `mesh_simplifier.bin`, a
  binary that is not in the repo, via `os.system`. Any use raises or silently produces nothing.
- **Load-time warnings that are noise**, not problems: the `torch_dtype` deprecation, *"no file named
  diffusion_pytorch_model.safetensors ... Defaulting to unsafe serialization"* (the paint VAE ships a `.bin`), and
  *"Expected types for unet ... got diffusers_modules.local.modules.UNet2p5DConditionModel"*.

## Verifying a change to the texture path

Fastest end-to-end check — textures an existing mesh, no shape generation, ~30 s on the turbo model:

```bash
source env/bin/activate
python3 -c "
import trimesh; from PIL import Image
from hy3dgen.texgen import Hunyuan3DPaintPipeline
p = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
m = p(trimesh.load('assets/1.glb', force='mesh'), image=Image.open('assets/demo.png'))
m.export('/tmp/check.glb')
print(type(m.visual.material).__name__, m.visual.material.baseColorTexture.size)
"   # expect: PBRMaterial (2048, 2048)
```

The full app path can be driven headlessly with `gradio_client` against a running `gradio_app.py`
(`api_name='/generation_all'` and `'/on_export_click'`), which is how fixes 1-3 were verified.

Reference timings, Quadro RTX 6000, `tencent/Hunyuan3D-2` + `hunyuan3d-dit-v2-0`, `--low_vram_mode`,
30 steps, octree 256, turbo paint: shape 27 s, face reduction 7 s, texture 32 s, **total ~67 s** per asset.
