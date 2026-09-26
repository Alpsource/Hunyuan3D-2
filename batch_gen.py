# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""Turn a folder of images into textured, game-ready GLB files, unattended.

Models are loaded once and reused for the whole folder. A failing asset is logged and skipped
rather than killing the run, and finished assets are recorded in manifest.csv as they complete,
so an interrupted run can simply be restarted - existing outputs are skipped by default.

    python3 batch_gen.py --input_dir images/ --output_dir out/

See --help for the knobs. Per-asset pipeline:
    background removal -> shape -> cleanup -> decimate -> texture -> out/<name>.glb
"""

import argparse
import csv
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

MANIFEST_FIELDS = [
    'name', 'input', 'status', 'seed', 'faces', 'vertices', 'textured',
    'shape_seconds', 'texture_seconds', 'total_seconds',
    'shape_model', 'paint_model', 'finished_at', 'error',
]


def find_images(input_dir, recursive):
    """Collect input images in a stable order."""
    walker = Path(input_dir).rglob('*') if recursive else Path(input_dir).glob('*')
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def assign_output_names(images, output_dir):
    """Map each image to a unique output stem. Recursive runs can collide on stem alone."""
    names, used = {}, set()
    for path in images:
        stem = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in path.stem).strip('._') or 'asset'
        name, n = stem, 1
        while name in used:
            name, n = f'{stem}_{n}', n + 1
        used.add(name)
        names[path] = os.path.join(output_dir, f'{name}.glb')
    return names


class ManifestWriter:
    """Append-only manifest, flushed per asset so a killed run keeps its history."""

    def __init__(self, path):
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.handle = open(path, 'a', newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
        if is_new:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        self.handle.close()


def load_image(path, rembg_worker):
    """Load an image and ensure it has an alpha channel the shape model can use."""
    image = Image.open(path)
    if image.mode == 'RGBA' and image.getextrema()[3][0] < 255:
        return image  # already cut out
    return rembg_worker(image.convert('RGB'))


def build_workers(args):
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import (DegenerateFaceRemover, FaceReducer, FloaterRemover,
                                  Hunyuan3DDiTFlowMatchingPipeline)

    workers = {
        'rembg': BackgroundRemover(),
        'floater': FloaterRemover(),
        'degenerate': DegenerateFaceRemover(),
        'reducer': FaceReducer(),
    }

    print(f'Loading shape model {args.model_path}/{args.subfolder} ...', flush=True)
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path, subfolder=args.subfolder, use_safetensors=True, device=args.device)
    if args.enable_flashvdm:
        shape.enable_flashvdm(mc_algo='mc' if args.device in ('cpu', 'mps') else args.mc_algo)
    if args.compile:
        shape.compile()
    workers['shape'] = shape

    if not args.no_texture:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        print(f'Loading paint model {args.texgen_model_path}/{args.texgen_subfolder} '
              f'(first use downloads several GB) ...', flush=True)
        paint = Hunyuan3DPaintPipeline.from_pretrained(
            args.texgen_model_path, subfolder=args.texgen_subfolder)
        if args.texture_size != paint.config.texture_size:
            # The renderer is built in the constructor, so resizing means rebuilding it.
            from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
            paint.config.texture_size = args.texture_size
            paint.render = MeshRender(default_resolution=paint.config.render_size,
                                      texture_size=args.texture_size)
        if args.low_vram_mode:
            paint.enable_model_cpu_offload()
        workers['paint'] = paint

    return workers


def generate_one(image_path, output_path, workers, args, seed):
    """Run one image through the pipeline. Returns a manifest row."""
    started = time.time()
    image = load_image(image_path, workers['rembg'])

    shape_started = time.time()
    mesh = workers['shape'](
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=torch.manual_seed(seed),
        output_type='trimesh',
    )[0]
    shape_seconds = time.time() - shape_started

    if not args.no_clean:
        mesh = workers['floater'](mesh)
        mesh = workers['degenerate'](mesh)
    if args.target_face_num > 0:
        mesh = workers['reducer'](mesh, max_facenum=args.target_face_num)

    texture_seconds = 0.0
    if 'paint' in workers:
        texture_started = time.time()
        # Decimate before painting: the UV unwrap and bake then run on the lighter mesh.
        mesh = workers['paint'](mesh, image=image)
        texture_seconds = time.time() - texture_started

    total_seconds = time.time() - started
    row = {
        'name': Path(output_path).stem,
        'input': str(image_path),
        'status': 'ok',
        'seed': seed,
        'faces': len(mesh.faces),
        'vertices': len(mesh.vertices),
        'textured': 'paint' in workers,
        'shape_seconds': round(shape_seconds, 1),
        'texture_seconds': round(texture_seconds, 1),
        'total_seconds': round(total_seconds, 1),
        'shape_model': f'{args.model_path}/{args.subfolder}',
        'paint_model': '' if args.no_texture else f'{args.texgen_model_path}/{args.texgen_subfolder}',
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'error': '',
    }
    mesh.metadata['extras'] = {k: row[k] for k in
                               ('seed', 'shape_model', 'paint_model', 'faces', 'input')}
    mesh.export(output_path, include_normals='paint' in workers)
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch image -> textured GLB generation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io_group = parser.add_argument_group('input / output')
    io_group.add_argument('--input_dir', type=str, required=True, help='Folder of input images.')
    io_group.add_argument('--output_dir', type=str, required=True, help='Where the .glb files go.')
    io_group.add_argument('--recursive', action='store_true', help='Also search sub-folders.')
    io_group.add_argument('--overwrite', action='store_true',
                          help='Regenerate assets that already exist (default: skip and resume).')
    io_group.add_argument('--limit', type=int, default=0,
                          help='Only process the first N pending images. 0 means all.')

    mesh_group = parser.add_argument_group('mesh')
    mesh_group.add_argument('--target_face_num', type=int, default=10000,
                            help='Decimate to this many faces. 0 disables decimation.')
    mesh_group.add_argument('--no_clean', action='store_true',
                            help='Skip floater and degenerate-face removal.')
    mesh_group.add_argument('--steps', type=int, default=50, help='Shape diffusion steps.')
    mesh_group.add_argument('--guidance_scale', type=float, default=5.0)
    mesh_group.add_argument('--octree_resolution', type=int, default=256,
                            help='Higher captures finer geometry and costs more VRAM.')
    mesh_group.add_argument('--num_chunks', type=int, default=200000)
    mesh_group.add_argument('--seed', type=int, default=1234)
    mesh_group.add_argument('--randomize_seed', action='store_true',
                            help='Use a fresh random seed per asset instead of --seed.')

    model_group = parser.add_argument_group('models')
    model_group.add_argument('--model_path', type=str, default='tencent/Hunyuan3D-2')
    model_group.add_argument('--subfolder', type=str, default='hunyuan3d-dit-v2-0')
    model_group.add_argument('--texgen_model_path', type=str, default='tencent/Hunyuan3D-2')
    model_group.add_argument('--texgen_subfolder', type=str, default='hunyuan3d-paint-v2-0',
                             choices=['hunyuan3d-paint-v2-0', 'hunyuan3d-paint-v2-0-turbo'],
                             help='Default is the full model; -turbo is faster and slightly lower quality.')
    model_group.add_argument('--texture_size', type=int, default=2048,
                             help='UV texture resolution. Drop to 1024 for mobile targets.')
    model_group.add_argument('--no_texture', action='store_true',
                             help='Shape only - skips loading the paint model entirely.')

    runtime_group = parser.add_argument_group('runtime')
    runtime_group.add_argument('--device', type=str, default='cuda')
    runtime_group.add_argument('--mc_algo', type=str, default='mc')
    runtime_group.add_argument('--low_vram_mode', action='store_true')
    runtime_group.add_argument('--enable_flashvdm', action='store_true')
    runtime_group.add_argument('--compile', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f'Input folder not found: {args.input_dir}')

    images = find_images(args.input_dir, args.recursive)
    if not images:
        sys.exit(f'No images ({", ".join(sorted(IMAGE_EXTENSIONS))}) found in {args.input_dir}')

    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    outputs = assign_output_names(images, args.output_dir)
    pending = [p for p in images if args.overwrite or not os.path.exists(outputs[p])]
    skipped = len(images) - len(pending)
    if args.limit > 0:
        pending = pending[:args.limit]

    print(f'{len(images)} image(s) found, {skipped} already done, {len(pending)} to generate.')
    if not pending:
        print('Nothing to do.')
        return

    workers = build_workers(args)
    manifest = ManifestWriter(os.path.join(args.output_dir, 'manifest.csv'))
    succeeded = failed = 0
    run_started = time.time()

    try:
        for index, image_path in enumerate(pending, start=1):
            output_path = outputs[image_path]
            name = Path(output_path).stem
            seed = random.randint(0, 2 ** 31 - 1) if args.randomize_seed else args.seed
            print(f'[{index}/{len(pending)}] {image_path.name} -> {name}.glb', flush=True)
            try:
                row = generate_one(image_path, output_path, workers, args, seed)
                succeeded += 1
                print(f'    ok: {row["faces"]} faces in {row["total_seconds"]}s', flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                failed += 1
                log_path = os.path.join(log_dir, f'{name}.log')
                with open(log_path, 'w') as handle:
                    traceback.print_exc(file=handle)
                print(f'    FAILED: {error}\n    traceback: {log_path}', file=sys.stderr, flush=True)
                row = {
                    'name': name, 'input': str(image_path), 'status': 'failed', 'seed': seed,
                    'finished_at': datetime.now().isoformat(timespec='seconds'),
                    'error': str(error)[:500],
                }
            manifest.write(row)
            if args.low_vram_mode:
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print('\nInterrupted. Re-run the same command to resume.', file=sys.stderr)
    finally:
        manifest.close()

    elapsed = time.time() - run_started
    print(f'\nDone: {succeeded} generated, {failed} failed, {skipped} skipped '
          f'in {elapsed / 60:.1f} min. Output: {args.output_dir}')
    if failed:
        print(f'Failures are listed in manifest.csv with tracebacks in {log_dir}')
        sys.exit(1)


if __name__ == '__main__':
    main()
