# This module is bpy script generate mesh using 3d feature reconstructed, 
# run this module after 3d_feature_recon.py
# Prerequisite: 
# 1. VS Code IDE
# 2. have a json file with AABB data for meshes
# 3. have Blender installed (projected used is version 4.4)
# 4. downloaded blender developer extension for run
    # Extension Identifier: jacqueslucke.blender-development
    # Version 0.0.26



import bpy
import json
from pathlib import Path

# Resolve path to JSON file relative to this script
script_dir = Path(__file__).resolve().parent  # ./blender/
json_path = script_dir.parent / "data" / "json" / "exported_aabb_only.json"

# Resolve save path for blend file
blend_dir = script_dir / "blend files"
blend_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists
blend_path = blend_dir / "aabb_cubes.blend" # aabb cubes = primitie cubes representing AABBs

# Load AABB data
with open(json_path, "r") as f:
    aabb_data = json.load(f)

# Optional: clear existing objects
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# Create cube for each AABB
def draw_aabb(name, min_coords, max_coords):
    # Compute center and size
    center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
    size = [(max_coords[i] - min_coords[i]) for i in range(3)]

    # Add cube at origin
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = f"AABB_{name}"
    obj.scale = [s / 2 for s in size]  # Blender cubes are 2x2x2 by default
    obj.location = center

# Main execution
clear_scene()

for mesh_name, mesh_info in aabb_data.items():
    aabb = mesh_info.get("aabb", {})
    min_coords = aabb.get("min", [0.0, 0.0, 0.0])
    max_coords = aabb.get("max", [0.0, 0.0, 0.0])
    draw_aabb(mesh_name, min_coords, max_coords)

# Save the blend file
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

print(f" AABB cubes created for all meshes. Blend file saved at: {blend_path}")
