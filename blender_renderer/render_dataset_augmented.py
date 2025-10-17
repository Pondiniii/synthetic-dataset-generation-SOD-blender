#!/usr/bin/env python3
"""
Augmented Renderer for SOD with High-res + Random Augmentations
- High resolution support (up to 4096x4096)
- JPEG XL lossless output
- Random HDR rotation per pitch angle
- Random HDR switching (different HDR per angle group)
- Random lighting intensity variation
- Smart cropping strategy (tight/medium/loose)
- Flat UUID-based structure
"""
import bpy
import math
import sys
import uuid
import random
import subprocess
from pathlib import Path
from typing import Optional
from mathutils import Vector, Euler


class AugmentedRenderer:
    def __init__(self, output_res=640, session_uuid=None, enable_augmentations=True):
        """
        Args:
            output_res: Output resolution (640, 1024, 2048, 4096)
            session_uuid: Session UUID (auto-generated if None)
            enable_augmentations: Enable random augmentations
        """
        self.output_res = output_res
        self.session_uuid = session_uuid or str(uuid.uuid4())[:8]
        self.enable_augmentations = enable_augmentations
        self.scene = bpy.context.scene
        self.setup_render_settings()

    def setup_render_settings(self):
        """Configure Cycles renderer"""
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

        # Performance settings based on resolution
        if self.output_res >= 2048:
            self.scene.cycles.samples = 128
            self.scene.cycles.use_denoising = True
            self.scene.cycles.denoiser = 'OPTIX'  # GPU denoising
        else:
            self.scene.cycles.samples = 64
            self.scene.cycles.use_denoising = False

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

    def normalize_object_size(self, obj, camera_strategy='medium'):
        """
        Normalize object to fit in unit cube and return optimal camera distance

        Args:
            camera_strategy: 'tight' (1.0x), 'medium' (1.2x), 'loose' (1.5x)

        Returns:
            float: Optimal camera distance based on object size
        """
        obj.location = (0, 0, 0)

        dims = obj.dimensions
        max_dim = max(dims.x, dims.y, dims.z)

        if max_dim > 0:
            scale_factor = 1.0 / max_dim
            obj.scale = (scale_factor, scale_factor, scale_factor)

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Calculate optimal camera distance with strategy
        base_distance = 3.5  # Increased for better framing

        if camera_strategy == 'tight':
            multiplier = 0.85
        elif camera_strategy == 'loose':
            multiplier = 1.3
        else:  # medium
            multiplier = 1.0

        optimal_distance = base_distance * multiplier
        return optimal_distance

    def setup_hdr_environment(self, hdr_path: Path, rotation_z=0.0, strength=1.0):
        """
        Setup HDR environment with rotation and strength

        Args:
            hdr_path: Path to HDR file
            rotation_z: Z-axis rotation in radians (0-2π)
            strength: Light intensity multiplier (0.5-2.0)
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

            # Mapping node for rotation
            mapping = nodes.new(type='ShaderNodeMapping')
            mapping.inputs['Rotation'].default_value = (0, 0, rotation_z)

            # Texture coordinate
            tex_coord = nodes.new(type='ShaderNodeTexCoord')

            # Environment texture
            env_tex = nodes.new(type='ShaderNodeTexEnvironment')
            env_tex.image = bpy.data.images.load(str(hdr_path))

            # Background shader with strength
            background = nodes.new(type='ShaderNodeBackground')
            background.inputs['Strength'].default_value = strength

            # Output
            output = nodes.new(type='ShaderNodeOutputWorld')

            # Connect
            links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
            links.new(env_tex.outputs['Color'], background.inputs['Color'])
            links.new(background.outputs['Background'], output.inputs['Surface'])

    def create_camera(self, distance=3.5):
        """Create camera at specified distance"""
        cam_data = bpy.data.cameras.new(name='Camera')
        cam_data.lens = 50

        cam_obj = bpy.data.objects.new('Camera', cam_data)
        self.scene.collection.objects.link(cam_obj)
        self.scene.camera = cam_obj

        cam_obj.location = (0, -distance, 0)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)

        return cam_obj

    def random_camera_positions(self, pitch_steps=10, yaw_per_pitch=4):
        """
        Generate random camera positions

        Args:
            pitch_steps: Number of pitch angles (0-90°)
            yaw_per_pitch: Random yaw angles per pitch

        Returns:
            List of (yaw, pitch) tuples
        """
        positions = []

        pitch_max = math.pi / 2  # 90 degrees
        pitch_increment = pitch_max / (pitch_steps - 1) if pitch_steps > 1 else 0

        for pitch_idx in range(pitch_steps):
            pitch = pitch_idx * pitch_increment

            # Generate random yaw angles for this pitch
            for _ in range(yaw_per_pitch):
                yaw = random.uniform(0, 2 * math.pi)
                positions.append((yaw, pitch))

        return positions

    def set_camera_spherical(self, camera, yaw, pitch, distance=3.5):
        """Set camera to spherical coordinates"""
        x = distance * math.cos(pitch) * math.sin(yaw)
        y = distance * math.cos(pitch) * math.cos(yaw)
        z = distance * math.sin(pitch)

        camera.location = (x, y, z)

        direction = Vector((0, 0, 0)) - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()

    def convert_to_jxl(self, png_path: Path, quality: Optional[int] = None) -> Path:
        """
        Convert PNG to JPEG XL with optional quality setting

        Args:
            png_path: Path to PNG file
            quality: JPEG XL quality (None = lossless)

        Returns:
            Path to JXL file
        """
        jxl_path = png_path.with_suffix('.jxl')

        try:
            # Build cjxl command; prefer quality when provided, otherwise use lossless
            cmd = ['cjxl', str(png_path), str(jxl_path), '-e', '7']
            if quality is None:
                cmd.extend(['-d', '0'])
            else:
                cmd.extend(['-q', str(quality)])

            subprocess.run(cmd, check=True, capture_output=True)
            png_path.unlink()  # Delete PNG
            return jxl_path
        except subprocess.CalledProcessError as e:
            print(f"⚠️  JPEG XL conversion failed: {e}")
            return png_path  # Keep PNG on failure

    def render_rgb_and_mask(self, rgb_path: Path, mask_path: Path, output_format='jxl'):
        """
        Render RGB with HDR + alpha mask

        Args:
            output_format: 'png' or 'jxl'
        """
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

        # Convert to JPEG XL if requested
        if output_format == 'jxl':
            rgb_path = self.convert_to_jxl(rgb_path, quality=80)
            mask_path = self.convert_to_jxl(mask_path, quality=99)

        return rgb_path, mask_path

    def render_dataset(self, model_path: Path, hdr_paths: list, output_dir: Path,
                      pitch_steps=10, yaw_per_pitch=4, hdr_switch_interval=4,
                      output_format='jxl'):
        """
        Render dataset with AUGMENTATIONS and FLAT STRUCTURE

        Args:
            model_path: Path to model .blend
            hdr_paths: List of HDR paths (switches between them)
            output_dir: Output directory
            pitch_steps: Number of pitch angles (0-90°)
            yaw_per_pitch: Random yaw angles per pitch
            hdr_switch_interval: Switch HDR every N images (0 = no switching)
            output_format: 'png' or 'jxl'

        Output structure:
            output_dir/
            ├── images/
            │   ├── {session}_{model}_{hdr}_{yaw}_{pitch}_{dist}_{rot}_{str}.jxl
            │   └── ...
            └── masks/
                ├── {session}_{model}_{hdr}_{yaw}_{pitch}_{dist}_{rot}_{str}.jxl
                └── ...
        """
        # Extract short UUIDs
        model_uuid = model_path.stem[:8]

        print(f"\n{'='*70}")
        print(f"🎬 AUGMENTED RENDERING ({output_format.upper()})")
        print(f"{'='*70}")
        print(f"📦 Model:      {model_path.name} ({model_uuid})")
        print(f"🌍 HDRs:       {len(hdr_paths)} environments")
        print(f"🔖 Session:    {self.session_uuid}")
        print(f"📐 Views:      {pitch_steps} pitch × {yaw_per_pitch} random yaw = {pitch_steps * yaw_per_pitch}")
        print(f"📏 Resolution: {self.output_res}×{self.output_res}")
        print(f"🎲 Augment:    {'✅ Enabled' if self.enable_augmentations else '❌ Disabled'}")
        print(f"📂 Output:     {output_dir}")
        print(f"{'='*70}\n")

        # Setup scene
        self.clear_scene()

        # Load model ONCE (scene reuse optimization!)
        model = self.load_model(model_path)
        if not model:
            print("❌ Failed to load model")
            return

        # Generate random camera positions
        positions = self.random_camera_positions(pitch_steps, yaw_per_pitch)
        random.shuffle(positions)  # Extra randomization

        # Create flat output directories
        rgb_dir = output_dir / "images"
        mask_dir = output_dir / "masks"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Render loop with scene reuse
        for idx, (yaw, pitch) in enumerate(positions):
            # Random augmentations per image
            if self.enable_augmentations:
                # Camera distance strategy (tight/medium/loose)
                camera_strategy = random.choice(['tight', 'medium', 'loose'])

                # HDR rotation (0-360°)
                hdr_rotation = random.uniform(0, 2 * math.pi)

                # Lighting intensity (0.7-1.5x)
                light_strength = random.uniform(0.7, 1.5)

                # HDR switching
                if hdr_switch_interval > 0 and len(hdr_paths) > 1:
                    hdr_idx = (idx // hdr_switch_interval) % len(hdr_paths)
                else:
                    hdr_idx = 0

                hdr_path = hdr_paths[hdr_idx]
            else:
                camera_strategy = 'medium'
                hdr_rotation = 0.0
                light_strength = 1.0
                hdr_path = hdr_paths[0]

            # Normalize object with strategy (done once per image for safety)
            optimal_distance = self.normalize_object_size(model, camera_strategy)

            # Setup HDR with augmentations (scene reuse - only HDR changes!)
            if idx == 0 or self.enable_augmentations:
                self.setup_hdr_environment(hdr_path, hdr_rotation, light_strength)

            # Create camera (only once)
            if idx == 0:
                camera = self.create_camera(distance=optimal_distance)
            else:
                # Reuse camera, just update position
                camera = self.scene.camera

            # Set camera position
            self.set_camera_spherical(camera, yaw, pitch, distance=optimal_distance)

            # Prepare filenames with augmentation metadata
            yaw_deg = int(math.degrees(yaw))
            pitch_deg = int(math.degrees(pitch))
            hdr_uuid = hdr_path.stem[:8]

            # Encode augmentations in filename
            dist_code = {'tight': 't', 'medium': 'm', 'loose': 'l'}[camera_strategy]
            rot_deg = int(math.degrees(hdr_rotation))
            str_code = int(light_strength * 100)  # 70-150

            filename = f"{self.session_uuid}_{model_uuid}_{hdr_uuid}_{yaw_deg:03d}_{pitch_deg:02d}_{dist_code}_{rot_deg:03d}_{str_code:03d}.png"
            rgb_path = rgb_dir / filename
            mask_path = mask_dir / filename

            print(f"[{idx+1}/{len(positions)}] Y:{yaw_deg:3d}° P:{pitch_deg:2d}° D:{dist_code} R:{rot_deg:3d}° S:{str_code} HDR:{hdr_idx} -> {filename}")

            self.render_rgb_and_mask(rgb_path, mask_path, output_format)

        print(f"\n{'='*70}")
        print(f"✅ DATASET COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Rendered:   {len(positions)} image pairs")
        print(f"📁 Images:     {rgb_dir}")
        print(f"📁 Masks:      {mask_dir}")
        print(f"💾 Format:     {output_format.upper()}")
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
        print("║         Augmented Renderer (High-res + JPEG XL)                 ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║ Usage:                                                           ║")
        print("║   blender --background --python render_dataset_augmented.py --   ║")
        print("║     <model.blend> <hdr1.exr,hdr2.exr,...> <output_dir>          ║")
        print("║     [resolution] [pitch_steps] [yaw_per_pitch]                   ║")
        print("║     [hdr_switch_interval] [output_format] [session_uuid]         ║")
        print("║                                                                  ║")
        print("║ Arguments:                                                       ║")
        print("║   model.blend         - Path to 3D model                        ║")
        print("║   hdr1.exr,hdr2...    - Comma-separated HDR paths               ║")
        print("║   output_dir          - Output directory                        ║")
        print("║   resolution          - 640|1024|2048|4096 (default: 640)       ║")
        print("║   pitch_steps         - Pitch angles (default: 10)              ║")
        print("║   yaw_per_pitch       - Random yaw per pitch (default: 4)       ║")
        print("║   hdr_switch_interval - Switch HDR every N images (default: 4)  ║")
        print("║   output_format       - png|jxl (default: jxl)                  ║")
        print("║   session_uuid        - Session UUID (default: auto)            ║")
        print("║                                                                  ║")
        print("║ Example:                                                         ║")
        print("║   blender --background --python render_dataset_augmented.py --   ║")
        print("║     model.blend hdr1.exr,hdr2.exr output/ 2048 10 4 4 jxl       ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        sys.exit(1)

    model_path = Path(args[0])

    # Parse HDR paths (comma-separated)
    hdr_paths_str = args[1]
    hdr_paths = [Path(p.strip()) for p in hdr_paths_str.split(',')]

    output_dir = Path(args[2])

    resolution = int(args[3]) if len(args) > 3 else 640
    pitch_steps = int(args[4]) if len(args) > 4 else 10
    yaw_per_pitch = int(args[5]) if len(args) > 5 else 4
    hdr_switch_interval = int(args[6]) if len(args) > 6 else 4
    output_format = args[7] if len(args) > 7 else 'jxl'
    session_uuid = args[8] if len(args) > 8 else None

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    for hdr_path in hdr_paths:
        if not hdr_path.exists():
            print(f"❌ HDR not found: {hdr_path}")
            sys.exit(1)

    # Enable augmentations if multiple parameters varied
    enable_augmentations = (len(hdr_paths) > 1 or hdr_switch_interval > 0)

    renderer = AugmentedRenderer(
        output_res=resolution,
        session_uuid=session_uuid,
        enable_augmentations=enable_augmentations
    )

    renderer.render_dataset(
        model_path,
        hdr_paths,
        output_dir,
        pitch_steps,
        yaw_per_pitch,
        hdr_switch_interval,
        output_format
    )


if __name__ == '__main__':
    main()
