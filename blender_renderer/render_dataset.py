#!/usr/bin/env python3
"""
Modern Blender Headless Renderer for SOD Dataset Generation
- Spherical camera rotation (yaw + pitch)
- CUDA/OptiX accelerated rendering
- Alpha channel masks for segmentation
- 640x640 output
"""
import bpy
import math
import sys
from pathlib import Path
from mathutils import Vector, Euler


class BlenderRenderer:
    def __init__(self, output_res=640):
        self.output_res = output_res
        self.scene = bpy.context.scene
        self.setup_render_settings()

    def setup_render_settings(self):
        """Configure Cycles renderer with GPU acceleration"""
        # Cycles engine
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'GPU'

        # Try OptiX first, fallback to CUDA
        prefs = bpy.context.preferences.addons['cycles'].preferences
        try:
            prefs.compute_device_type = 'OPTIX'
            print("✅ Using OptiX")
        except:
            prefs.compute_device_type = 'CUDA'
            print("✅ Using CUDA")

        # Enable only GPU devices (not CPU)
        prefs.get_devices()
        for device in prefs.devices:
            # Only enable GPU, skip CPU
            if device.type != 'CPU':
                device.use = True
                print(f"   GPU: {device.name}")
            else:
                device.use = False

        # Resolution
        self.scene.render.resolution_x = self.output_res
        self.scene.render.resolution_y = self.output_res
        self.scene.render.resolution_percentage = 100

        # Performance settings
        self.scene.cycles.samples = 64  # Lower for speed + less VRAM
        self.scene.cycles.use_denoising = False  # Disable to save VRAM

        # Tile size for GPU
        self.scene.cycles.tile_size = 256

        # Film settings for alpha
        self.scene.render.film_transparent = True
        self.scene.render.image_settings.color_mode = 'RGBA'
        self.scene.render.image_settings.file_format = 'PNG'
        self.scene.render.image_settings.color_depth = '8'

    def clear_scene(self):
        """Remove default objects"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def load_model(self, blend_path: Path):
        """Load model from .blend file"""
        # Append all objects from the file
        with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
            data_to.objects = data_from.objects

        # Add to scene and get main object
        objects = []
        for obj in data_to.objects:
            if obj:
                self.scene.collection.objects.link(obj)
                objects.append(obj)

        # Find mesh object (skip cameras, lights, etc)
        model = None
        for obj in objects:
            if obj.type == 'MESH':
                model = obj
                break

        if not model:
            # If no mesh, take first object
            model = objects[0] if objects else None

        return model

    def normalize_object_size(self, obj):
        """Normalize object to fit in unit cube, centered at origin"""
        # Center at origin
        obj.location = (0, 0, 0)

        # Get dimensions
        dims = obj.dimensions
        max_dim = max(dims.x, dims.y, dims.z)

        if max_dim > 0:
            # Scale so largest dimension = 1
            scale_factor = 1.0 / max_dim
            obj.scale = (scale_factor, scale_factor, scale_factor)

        # Apply transforms
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        print(f"📏 Normalized object: {dims} -> {obj.dimensions}")

    def setup_hdr_environment(self, hdr_path: Path):
        """
        Setup HDR as world environment
        Supports both .hdr/.exr images and .blend files with World
        """
        if hdr_path.suffix == '.blend':
            # Load World from .blend file (BlenderKit format)
            with bpy.data.libraries.load(str(hdr_path), link=False) as (data_from, data_to):
                # Get first World
                if data_from.worlds:
                    data_to.worlds = [data_from.worlds[0]]

            if data_to.worlds:
                self.scene.world = data_to.worlds[0]
                print(f"🌍 HDR environment (from .blend): {hdr_path.name}")
            else:
                print(f"⚠️  No World found in {hdr_path.name}, using default")
        else:
            # Load image HDR (.hdr, .exr, .jpg)
            world = self.scene.world
            world.use_nodes = True
            nodes = world.node_tree.nodes
            links = world.node_tree.links

            # Clear existing nodes
            nodes.clear()

            # Environment texture
            env_tex = nodes.new(type='ShaderNodeTexEnvironment')
            env_tex.image = bpy.data.images.load(str(hdr_path))

            # Background shader
            background = nodes.new(type='ShaderNodeBackground')
            background.inputs['Strength'].default_value = 1.0

            # Output
            output = nodes.new(type='ShaderNodeOutputWorld')

            # Connect
            links.new(env_tex.outputs['Color'], background.inputs['Color'])
            links.new(background.outputs['Background'], output.inputs['Surface'])

            print(f"🌍 HDR environment (image): {hdr_path.name}")

    def create_camera(self, distance=3.0):
        """Create camera at specified distance"""
        cam_data = bpy.data.cameras.new(name='Camera')
        cam_data.lens = 50  # 50mm focal length

        cam_obj = bpy.data.objects.new('Camera', cam_data)
        self.scene.collection.objects.link(cam_obj)
        self.scene.camera = cam_obj

        # Initial position (looking at origin from distance)
        cam_obj.location = (0, -distance, 0)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)

        return cam_obj

    def spherical_camera_positions(self, yaw_steps=12, pitch_steps=8):
        """
        Generate spherical camera positions

        yaw_steps: 360/yaw_steps = angle increment (12 = every 30°)
        pitch_steps: from 0° to 90° (8 = every ~11°)

        Returns list of (yaw, pitch) tuples in radians
        """
        positions = []

        yaw_increment = 2 * math.pi / yaw_steps  # Full circle
        pitch_max = math.pi / 2  # 90 degrees
        pitch_increment = pitch_max / (pitch_steps - 1) if pitch_steps > 1 else 0

        for pitch_idx in range(pitch_steps):
            pitch = pitch_idx * pitch_increment

            for yaw_idx in range(yaw_steps):
                yaw = yaw_idx * yaw_increment
                positions.append((yaw, pitch))

        return positions

    def set_camera_spherical(self, camera, yaw, pitch, distance=3.0):
        """
        Set camera to spherical coordinates

        yaw: rotation around Z axis (0 to 2π)
        pitch: elevation angle (0 to π/2, 0=horizontal, π/2=top-down)
        distance: distance from origin
        """
        # Convert spherical to cartesian
        x = distance * math.cos(pitch) * math.sin(yaw)
        y = distance * math.cos(pitch) * math.cos(yaw)
        z = distance * math.sin(pitch)

        camera.location = (x, y, z)

        # Point camera at origin
        direction = Vector((0, 0, 0)) - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()

    def render_image(self, output_path: Path):
        """Render to file"""
        self.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

    def render_rgb_and_mask(self, rgb_path: Path, mask_path: Path):
        """Render RGB with HDR background + alpha mask (0-255 binary)

        Args:
            rgb_path: Output path for RGB image with HDR background
            mask_path: Output path for alpha mask (0-255 grayscale)
        """
        # 1. Render RGB with HDR background
        self.scene.render.film_transparent = False  # Show HDR background
        self.scene.render.image_settings.color_mode = 'RGB'
        self.scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)

        # 2. Render RGBA with transparent background to get alpha mask
        self.scene.render.film_transparent = True  # Transparent background
        self.scene.render.image_settings.color_mode = 'RGBA'

        # Render to temporary RGBA file
        temp_path = mask_path.parent / f"_temp_rgba_{mask_path.name}"
        self.scene.render.filepath = str(temp_path)
        bpy.ops.render.render(write_still=True)

        # Extract alpha channel using PIL
        try:
            from PIL import Image
            import numpy as np

            # Load RGBA
            img = Image.open(str(temp_path))

            # Extract alpha channel (4th channel)
            alpha = np.array(img.split()[3])

            # Save as grayscale (0-255)
            mask_img = Image.fromarray(alpha, mode='L')
            mask_img.save(str(mask_path))

            # Remove temp file
            temp_path.unlink()
        except Exception as e:
            print(f"⚠️  Failed to extract alpha: {e}")
            # Fallback: just rename RGBA to mask
            import shutil
            shutil.move(str(temp_path), str(mask_path))

    def render_dataset(self, model_path: Path, hdr_path: Path, output_dir: Path,
                      yaw_steps=12, pitch_steps=8, dataset_name=None):
        """
        Full rendering pipeline

        Args:
            model_path: Path to .blend model
            hdr_path: Path to HDR image
            output_dir: Output directory
            yaw_steps: Number of yaw angles (360° / steps)
            pitch_steps: Number of pitch angles (0-90°)
            dataset_name: Optional custom dataset name (default: model_hdr)
        """
        # Get clean names (truncate UUIDs to 8 chars for readability)
        model_id = model_path.stem[:8] if len(model_path.stem) > 8 else model_path.stem
        hdr_id = hdr_path.stem[:8] if len(hdr_path.stem) > 8 else hdr_path.stem

        # Create dataset subdirectory
        if dataset_name is None:
            dataset_name = f"{model_id}_{hdr_id}"

        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"🎬 RENDERING SOD DATASET")
        print(f"{'='*70}")
        print(f"📦 Model:    {model_path.name}")
        print(f"🌍 HDR:      {hdr_path.name}")
        print(f"📐 Views:    {yaw_steps} yaw × {pitch_steps} pitch = {yaw_steps * pitch_steps} renders")
        print(f"📂 Dataset:  {dataset_dir}")
        print(f"{'='*70}\n")

        # Setup scene
        self.clear_scene()

        # Load and normalize model
        model = self.load_model(model_path)
        if not model:
            print("❌ Failed to load model")
            return

        self.normalize_object_size(model)

        # Setup HDR
        if hdr_path.exists():
            self.setup_hdr_environment(hdr_path)

        # Create camera
        camera = self.create_camera(distance=3.0)

        # Generate camera positions
        positions = self.spherical_camera_positions(yaw_steps, pitch_steps)

        # Create subdirectories for RGB and masks
        rgb_dir = dataset_dir / "images"
        mask_dir = dataset_dir / "masks"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for idx, (yaw, pitch) in enumerate(positions):
            yaw_deg = int(math.degrees(yaw))
            pitch_deg = int(math.degrees(pitch))

            # Set camera position
            self.set_camera_spherical(camera, yaw, pitch, distance=3.0)

            # Clean filename: view_yaw_pitch.png
            filename = f"view_{yaw_deg:03d}_{pitch_deg:02d}.png"
            rgb_path = rgb_dir / filename
            mask_path = mask_dir / filename

            print(f"[{idx+1}/{len(positions)}] Yaw:{yaw_deg:3d}° Pitch:{pitch_deg:2d}° -> {filename}")

            self.render_rgb_and_mask(rgb_path, mask_path)

        print(f"\n{'='*70}")
        print(f"✅ DATASET COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Rendered:   {len(positions)} image pairs")
        print(f"📁 Images:     {rgb_dir}")
        print(f"📁 Masks:      {mask_dir}")
        print(f"{'='*70}\n")


def main():
    """CLI interface for rendering SOD datasets"""
    # Parse args (after --)
    try:
        argv_idx = sys.argv.index('--') + 1
        args = sys.argv[argv_idx:]
    except ValueError:
        args = sys.argv[1:]

    # Filter out empty strings from bash
    args = [arg for arg in args if arg]

    # Show usage
    if len(args) < 3:
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║         Blender SOD Dataset Renderer                            ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║ Usage:                                                           ║")
        print("║   blender --background --python render_dataset.py --             ║")
        print("║     <model.blend> <hdr.exr> <output_dir>                         ║")
        print("║     [yaw_steps] [pitch_steps] [dataset_name]                     ║")
        print("║                                                                  ║")
        print("║ Arguments:                                                       ║")
        print("║   model.blend     - Path to 3D model file                       ║")
        print("║   hdr.exr         - Path to HDR environment file                ║")
        print("║   output_dir      - Output directory for datasets               ║")
        print("║   yaw_steps       - Number of yaw angles (default: 12)          ║")
        print("║   pitch_steps     - Number of pitch angles (default: 8)         ║")
        print("║   dataset_name    - Custom dataset name (optional)              ║")
        print("║                                                                  ║")
        print("║ Example:                                                         ║")
        print("║   blender --background --python render_dataset.py --             ║")
        print("║     model.blend hdr.exr datasets/ 12 8                           ║")
        print("║                                                                  ║")
        print("║ Output Structure:                                                ║")
        print("║   datasets/{dataset_name}/images/view_000_00.png                 ║")
        print("║   datasets/{dataset_name}/masks/view_000_00.png                  ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        sys.exit(1)

    # Parse arguments
    model_path = Path(args[0])
    hdr_path = Path(args[1])
    output_dir = Path(args[2])

    yaw_steps = int(args[3]) if len(args) > 3 else 12
    pitch_steps = int(args[4]) if len(args) > 4 else 8
    dataset_name = args[5] if len(args) > 5 else None

    # Validate paths
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    if not hdr_path.exists():
        print(f"❌ HDR file not found: {hdr_path}")
        sys.exit(1)

    # Render dataset
    renderer = BlenderRenderer(output_res=640)
    renderer.render_dataset(
        model_path,
        hdr_path,
        output_dir,
        yaw_steps,
        pitch_steps,
        dataset_name
    )


if __name__ == '__main__':
    main()
