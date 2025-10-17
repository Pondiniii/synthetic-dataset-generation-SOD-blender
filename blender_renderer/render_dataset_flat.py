#!/usr/bin/env python3
"""
Flat Structure Renderer for SOD
- All images in single images/ folder
- All masks in single masks/ folder
- UUID-based naming: {session}_{model}_{hdr}_{yaw}_{pitch}.png
"""
import bpy
import math
import sys
import uuid
from pathlib import Path
from mathutils import Vector, Euler


class FlatStructureRenderer:
    def __init__(self, output_res=640, session_uuid=None):
        """
        Args:
            output_res: Output resolution
            session_uuid: Session UUID (auto-generated if None)
        """
        self.output_res = output_res
        self.session_uuid = session_uuid or str(uuid.uuid4())[:8]  # Short 8-char UUID
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

        # Performance settings
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

    def normalize_object_size(self, obj):
        """
        Normalize object to fit in unit cube and return optimal camera distance

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

        # Calculate optimal camera distance
        # After normalization, object fits in 1x1x1 cube
        # Camera FOV ~50mm lens needs distance ~2.5-3.0 for full frame
        # Add 20% margin to ensure object is fully visible
        optimal_distance = 3.0  # Safe default for normalized objects

        return optimal_distance

    def setup_hdr_environment(self, hdr_path: Path):
        """Setup HDR environment"""
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
                      yaw_steps=12, pitch_steps=8):
        """
        Render dataset with FLAT STRUCTURE and UUID naming

        Output structure:
            output_dir/
            ├── images/
            │   ├── {session}_{model}_{hdr}_{yaw}_{pitch}.png
            │   └── ...
            └── masks/
                ├── {session}_{model}_{hdr}_{yaw}_{pitch}.png
                └── ...
        """
        # Extract short UUIDs from filenames
        model_uuid = model_path.stem[:8]  # First 8 chars
        hdr_uuid = hdr_path.stem[:8]

        print(f"\n{'='*70}")
        print(f"🎬 FLAT STRUCTURE RENDERING")
        print(f"{'='*70}")
        print(f"📦 Model:      {model_path.name} ({model_uuid})")
        print(f"🌍 HDR:        {hdr_path.name} ({hdr_uuid})")
        print(f"🔖 Session:    {self.session_uuid}")
        print(f"📐 Views:      {yaw_steps} yaw × {pitch_steps} pitch = {yaw_steps * pitch_steps}")
        print(f"📂 Output:     {output_dir}")
        print(f"{'='*70}\n")

        # Setup scene
        self.clear_scene()

        # Load model
        model = self.load_model(model_path)
        if not model:
            print("❌ Failed to load model")
            return

        # Normalize and get optimal distance
        optimal_distance = self.normalize_object_size(model)
        print(f"📏 Camera distance: {optimal_distance:.2f}")

        # Setup HDR
        if hdr_path.exists():
            self.setup_hdr_environment(hdr_path)

        # Create camera with optimal distance
        camera = self.create_camera(distance=optimal_distance)

        # Generate camera positions
        positions = self.spherical_camera_positions(yaw_steps, pitch_steps)

        # Create flat output directories
        rgb_dir = output_dir / "images"
        mask_dir = output_dir / "masks"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for idx, (yaw, pitch) in enumerate(positions):
            yaw_deg = int(math.degrees(yaw))
            pitch_deg = int(math.degrees(pitch))

            # Set camera position with optimal distance
            self.set_camera_spherical(camera, yaw, pitch, distance=optimal_distance)

            # UUID-based filename: {session}_{model}_{hdr}_{yaw}_{pitch}.png
            filename = f"{self.session_uuid}_{model_uuid}_{hdr_uuid}_{yaw_deg:03d}_{pitch_deg:02d}.png"
            rgb_path = rgb_dir / filename
            mask_path = mask_dir / filename

            print(f"[{idx+1}/{len(positions)}] Y:{yaw_deg:3d}° P:{pitch_deg:2d}° -> {filename}")

            self.render_rgb_and_mask(rgb_path, mask_path)

        print(f"\n{'='*70}")
        print(f"✅ DATASET COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Rendered:   {len(positions)} image pairs")
        print(f"📁 Images:     {rgb_dir}")
        print(f"📁 Masks:      {mask_dir}")
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
        print("║         Flat Structure Renderer (UUID Naming)                   ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║ Usage:                                                           ║")
        print("║   blender --background --python render_dataset_flat.py --        ║")
        print("║     <model.blend> <hdr.exr> <output_dir>                         ║")
        print("║     [yaw_steps] [pitch_steps] [session_uuid]                     ║")
        print("║                                                                  ║")
        print("║ Arguments:                                                       ║")
        print("║   model.blend    - Path to 3D model                             ║")
        print("║   hdr.exr        - Path to HDR environment                      ║")
        print("║   output_dir     - Output directory                             ║")
        print("║   yaw_steps      - Number of yaw angles (default: 12)           ║")
        print("║   pitch_steps    - Number of pitch angles (default: 8)          ║")
        print("║   session_uuid   - Session UUID (default: auto-generated)       ║")
        print("║                                                                  ║")
        print("║ Output Structure:                                                ║")
        print("║   output_dir/images/{session}_{model}_{hdr}_{yaw}_{pitch}.png   ║")
        print("║   output_dir/masks/{session}_{model}_{hdr}_{yaw}_{pitch}.png    ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        sys.exit(1)

    model_path = Path(args[0])
    hdr_path = Path(args[1])
    output_dir = Path(args[2])

    yaw_steps = int(args[3]) if len(args) > 3 else 12
    pitch_steps = int(args[4]) if len(args) > 4 else 8
    session_uuid = args[5] if len(args) > 5 else None

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    if not hdr_path.exists():
        print(f"❌ HDR not found: {hdr_path}")
        sys.exit(1)

    renderer = FlatStructureRenderer(output_res=640, session_uuid=session_uuid)
    renderer.render_dataset(model_path, hdr_path, output_dir, yaw_steps, pitch_steps)


if __name__ == '__main__':
    main()
