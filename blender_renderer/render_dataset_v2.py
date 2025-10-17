#!/usr/bin/env python3
"""
Advanced Blender Renderer for SOD - Phase 2
Features:
- High resolution (4096x4096) for quality crops
- HDR rotation augmentation
- Lighting intensity variation
- Random camera distance (tight/medium/loose crops)
- OptiX denoising for high-res
"""
import bpy
import math
import sys
import random
from pathlib import Path
from mathutils import Vector, Euler


class AdvancedBlenderRenderer:
    def __init__(self, output_res=4096, enable_augmentations=True):
        """
        Args:
            output_res: Output resolution (default: 4096 for high-quality crops)
            enable_augmentations: Enable random augmentations during rendering
        """
        self.output_res = output_res
        self.enable_augmentations = enable_augmentations
        self.scene = bpy.context.scene
        self.setup_render_settings()

    def setup_render_settings(self):
        """Configure Cycles renderer with GPU acceleration for high-res"""
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

        # Enable only GPU devices
        prefs.get_devices()
        for device in prefs.devices:
            if device.type != 'CPU':
                device.use = True
                print(f"   GPU: {device.name}")
            else:
                device.use = False

        # Resolution
        self.scene.render.resolution_x = self.output_res
        self.scene.render.resolution_y = self.output_res
        self.scene.render.resolution_percentage = 100

        # Performance settings (optimized for high-res)
        if self.output_res >= 2048:
            # High-res settings
            self.scene.cycles.samples = 128
            self.scene.cycles.use_denoising = True  # Enable OptiX denoiser
            print(f"📐 High-res mode: {self.output_res}x{self.output_res}, 128 samples, denoising ON")
        else:
            # Low-res settings
            self.scene.cycles.samples = 64
            self.scene.cycles.use_denoising = False
            print(f"📐 Standard mode: {self.output_res}x{self.output_res}, 64 samples")

        # Tile size for GPU
        self.scene.cycles.tile_size = 256

        # Film settings
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
        with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
            data_to.objects = data_from.objects

        objects = []
        for obj in data_to.objects:
            if obj:
                self.scene.collection.objects.link(obj)
                objects.append(obj)

        # Find mesh object
        model = None
        for obj in objects:
            if obj.type == 'MESH':
                model = obj
                break

        if not model:
            model = objects[0] if objects else None

        return model

    def normalize_object_size(self, obj, scale=1.0):
        """
        Normalize object to fit in unit cube

        Args:
            scale: Scale factor (0.8 = tight crop, 1.0 = medium, 1.2 = loose)
        """
        obj.location = (0, 0, 0)

        dims = obj.dimensions
        max_dim = max(dims.x, dims.y, dims.z)

        if max_dim > 0:
            scale_factor = scale / max_dim
            obj.scale = (scale_factor, scale_factor, scale_factor)

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        print(f"📏 Normalized (scale={scale:.2f}): {dims} -> {obj.dimensions}")

    def setup_hdr_environment(self, hdr_path: Path, rotation_z=0.0, strength=1.0):
        """
        Setup HDR environment with rotation and strength augmentation

        Args:
            hdr_path: Path to HDR file
            rotation_z: Z-axis rotation in degrees (0-360)
            strength: Light strength multiplier (0.5-2.0)
        """
        if hdr_path.suffix == '.blend':
            # Load World from .blend
            with bpy.data.libraries.load(str(hdr_path), link=False) as (data_from, data_to):
                if data_from.worlds:
                    data_to.worlds = [data_from.worlds[0]]

            if data_to.worlds:
                self.scene.world = data_to.worlds[0]
        else:
            # Load image HDR
            world = self.scene.world
            world.use_nodes = True
            nodes = world.node_tree.nodes
            links = world.node_tree.links

            nodes.clear()

            # Texture coordinate (for rotation)
            tex_coord = nodes.new(type='ShaderNodeTexCoord')

            # Mapping node (for rotation)
            mapping = nodes.new(type='ShaderNodeMapping')
            mapping.inputs['Rotation'].default_value[2] = math.radians(rotation_z)

            # Environment texture
            env_tex = nodes.new(type='ShaderNodeTexEnvironment')
            env_tex.image = bpy.data.images.load(str(hdr_path))

            # Background shader
            background = nodes.new(type='ShaderNodeBackground')
            background.inputs['Strength'].default_value = strength

            # Output
            output = nodes.new(type='ShaderNodeOutputWorld')

            # Connect
            links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
            links.new(env_tex.outputs['Color'], background.inputs['Color'])
            links.new(background.outputs['Background'], output.inputs['Surface'])

        print(f"🌍 HDR: {hdr_path.name} (rot={rotation_z:.0f}°, str={strength:.2f})")

    def create_camera(self, distance=3.0):
        """Create camera at specified distance"""
        cam_data = bpy.data.cameras.new(name='Camera')
        cam_data.lens = 50

        cam_obj = bpy.data.objects.new('Camera', cam_data)
        self.scene.collection.objects.link(cam_obj)
        self.scene.camera = cam_obj

        cam_obj.location = (0, -distance, 0)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)

        return cam_obj

    def spherical_camera_positions(self, yaw_steps=12, pitch_steps=8):
        """Generate spherical camera positions"""
        positions = []

        yaw_increment = 2 * math.pi / yaw_steps
        pitch_max = math.pi / 2
        pitch_increment = pitch_max / (pitch_steps - 1) if pitch_steps > 1 else 0

        for pitch_idx in range(pitch_steps):
            pitch = pitch_idx * pitch_increment

            for yaw_idx in range(yaw_steps):
                yaw = yaw_idx * yaw_increment
                positions.append((yaw, pitch))

        return positions

    def set_camera_spherical(self, camera, yaw, pitch, distance=3.0):
        """Set camera to spherical coordinates"""
        x = distance * math.cos(pitch) * math.sin(yaw)
        y = distance * math.cos(pitch) * math.cos(yaw)
        z = distance * math.sin(pitch)

        camera.location = (x, y, z)

        direction = Vector((0, 0, 0)) - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()

    def render_rgb_and_mask(self, rgb_path: Path, mask_path: Path):
        """Render RGB with HDR + alpha mask"""
        # 1. RGB with HDR
        self.scene.render.film_transparent = False
        self.scene.render.image_settings.color_mode = 'RGB'
        self.scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)

        # 2. RGBA for mask
        self.scene.render.film_transparent = True
        self.scene.render.image_settings.color_mode = 'RGBA'

        temp_path = mask_path.parent / f"_temp_rgba_{mask_path.name}"
        self.scene.render.filepath = str(temp_path)
        bpy.ops.render.render(write_still=True)

        # Extract alpha
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(str(temp_path))
            alpha = np.array(img.split()[3])
            mask_img = Image.fromarray(alpha, mode='L')
            mask_img.save(str(mask_path))
            temp_path.unlink()
        except Exception as e:
            print(f"⚠️  Alpha extraction failed: {e}")
            import shutil
            shutil.move(str(temp_path), str(mask_path))

    def render_dataset(self, model_path: Path, hdr_path: Path, output_dir: Path,
                      yaw_steps=12, pitch_steps=10, dataset_name=None):
        """
        Full rendering pipeline with Phase 2 augmentations

        Args:
            model_path: Path to .blend model
            hdr_path: Path to HDR
            output_dir: Output directory
            yaw_steps: Yaw angles (default: 12)
            pitch_steps: Pitch angles (default: 10 for 0°-90° every 10°)
            dataset_name: Custom name
        """
        # Create dataset name
        model_id = model_path.stem[:8]
        hdr_id = hdr_path.stem[:8]

        if dataset_name is None:
            dataset_name = f"{model_id}_{hdr_id}"

        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"🎬 PHASE 2 RENDERING (HIGH-RES + AUGMENTATIONS)")
        print(f"{'='*70}")
        print(f"📦 Model:       {model_path.name}")
        print(f"🌍 HDR:         {hdr_path.name}")
        print(f"📐 Resolution:  {self.output_res}x{self.output_res}")
        print(f"📐 Views:       {yaw_steps} yaw × {pitch_steps} pitch = {yaw_steps * pitch_steps}")
        print(f"🎲 Augment:     {'ON' if self.enable_augmentations else 'OFF'}")
        print(f"📂 Dataset:     {dataset_dir}")
        print(f"{'='*70}\n")

        # Setup scene
        self.clear_scene()

        # Load model
        model = self.load_model(model_path)
        if not model:
            print("❌ Failed to load model")
            return

        # Generate camera positions
        positions = self.spherical_camera_positions(yaw_steps, pitch_steps)

        # Create output dirs
        rgb_dir = dataset_dir / "images"
        mask_dir = dataset_dir / "masks"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Create camera (distance will vary)
        camera = self.create_camera(distance=3.0)

        for idx, (yaw, pitch) in enumerate(positions):
            yaw_deg = int(math.degrees(yaw))
            pitch_deg = int(math.degrees(pitch))

            # AUGMENTATION 1: Random camera distance (crop tightness)
            if self.enable_augmentations:
                # 40% tight, 40% medium, 20% loose
                rand = random.random()
                if rand < 0.4:
                    distance = 2.5  # Tight crop
                    scale = 0.9
                elif rand < 0.8:
                    distance = 3.0  # Medium
                    scale = 1.0
                else:
                    distance = 3.8  # Loose crop
                    scale = 1.1
            else:
                distance = 3.0
                scale = 1.0

            # Normalize with scale
            self.normalize_object_size(model, scale=scale)

            # AUGMENTATION 2: HDR rotation per pitch level
            if self.enable_augmentations:
                # Rotate HDR differently for each pitch level
                hdr_rotation = (pitch_deg * 37) % 360  # Deterministic but varied
                # Add small random component
                hdr_rotation += random.uniform(-10, 10)
            else:
                hdr_rotation = 0

            # AUGMENTATION 3: Random lighting intensity
            if self.enable_augmentations:
                strength = random.uniform(0.7, 1.5)
            else:
                strength = 1.0

            # Setup HDR (only once per pitch level for efficiency)
            if idx % yaw_steps == 0:  # First yaw of each pitch level
                self.setup_hdr_environment(hdr_path, rotation_z=hdr_rotation, strength=strength)

            # Set camera position
            self.set_camera_spherical(camera, yaw, pitch, distance=distance)

            # Render
            filename = f"view_{yaw_deg:03d}_{pitch_deg:02d}.png"
            rgb_path = rgb_dir / filename
            mask_path = mask_dir / filename

            aug_str = f"d={distance:.1f} rot={hdr_rotation:.0f}° str={strength:.2f}" if self.enable_augmentations else ""
            print(f"[{idx+1}/{len(positions)}] Y:{yaw_deg:3d}° P:{pitch_deg:2d}° {aug_str:30s} -> {filename}")

            self.render_rgb_and_mask(rgb_path, mask_path)

        print(f"\n{'='*70}")
        print(f"✅ PHASE 2 DATASET COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Rendered:    {len(positions)} image pairs")
        print(f"📐 Resolution:  {self.output_res}x{self.output_res}")
        print(f"📁 Images:      {rgb_dir}")
        print(f"📁 Masks:       {mask_dir}")
        print(f"{'='*70}\n")


def main():
    """CLI interface"""
    try:
        argv_idx = sys.argv.index('--') + 1
        args = sys.argv[argv_idx:]
    except ValueError:
        args = sys.argv[1:]

    args = [arg for arg in args if arg]

    if len(args) < 3:
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║         Phase 2 Renderer (High-Res + Augmentations)             ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║ Usage:                                                           ║")
        print("║   blender --background --python render_dataset_v2.py --          ║")
        print("║     <model.blend> <hdr.exr> <output_dir>                         ║")
        print("║     [resolution] [yaw] [pitch] [augment]                         ║")
        print("║                                                                  ║")
        print("║ Arguments:                                                       ║")
        print("║   model.blend  - Path to 3D model                               ║")
        print("║   hdr.exr      - Path to HDR environment                        ║")
        print("║   output_dir   - Output directory                               ║")
        print("║   resolution   - Output resolution (default: 4096)              ║")
        print("║   yaw          - Yaw steps (default: 12)                        ║")
        print("║   pitch        - Pitch steps (default: 10)                      ║")
        print("║   augment      - Enable augmentations: 1=ON, 0=OFF (default: 1)║")
        print("║                                                                  ║")
        print("║ Examples:                                                        ║")
        print("║   # High-res with augmentations                                 ║")
        print("║   blender --background --python render_dataset_v2.py --          ║")
        print("║     model.blend hdr.exr datasets/ 4096 12 10 1                   ║")
        print("║                                                                  ║")
        print("║   # Standard res without augmentations                          ║")
        print("║   blender --background --python render_dataset_v2.py --          ║")
        print("║     model.blend hdr.exr datasets/ 640 12 8 0                     ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        sys.exit(1)

    model_path = Path(args[0])
    hdr_path = Path(args[1])
    output_dir = Path(args[2])

    resolution = int(args[3]) if len(args) > 3 else 4096
    yaw_steps = int(args[4]) if len(args) > 4 else 12
    pitch_steps = int(args[5]) if len(args) > 5 else 10
    enable_augmentations = bool(int(args[6])) if len(args) > 6 else True

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    if not hdr_path.exists():
        print(f"❌ HDR not found: {hdr_path}")
        sys.exit(1)

    renderer = AdvancedBlenderRenderer(
        output_res=resolution,
        enable_augmentations=enable_augmentations
    )

    renderer.render_dataset(
        model_path,
        hdr_path,
        output_dir,
        yaw_steps,
        pitch_steps
    )


if __name__ == '__main__':
    main()
