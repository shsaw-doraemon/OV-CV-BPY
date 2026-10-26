# This module contains:
# 1. annotation preprocessing function
# 2. Image-to-cartesian coordinates mapping function
# 3. 3D AABB Reconstruction function
# 4. visualization functions (drawing graphs)
# 5. module testing functions

from pathlib import Path # used in 1.0, 2.0, 3.0
import numpy as np # used in 2.0, 3.0
import matplotlib.pyplot as plt # used in 2.0, 3.0
import json # used in 1.0, 2.0, 3.0
import numpy as np # used in 2.0, 3.0
from functools import partial # used in 2.0
from collections import defaultdict # used in 1.0, 3.0
import matplotlib.pyplot as plt # used in 2.0, 3.0


#1.0 Annotations_Processing 
def preprocess_annotations(json_path):
    """
    this function tracks unique meshs from views and store the segmentations in mesh_dict
    """
    # Nested defaultdict structure
    mesh_dict = defaultdict(lambda: {
        "views": set(),
        "segmentations": defaultdict(list),
        "aabb": { "min": [float('inf')] * 3, "max": [float('-inf')] * 3 }  # Placeholder for AABB
    })

    # Load annotations
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return {}

    annotations = data.get("annotations", [])
    print(f"🔍 Loaded {len(annotations)} annotations")

    for i, entry in enumerate(annotations):
        try:
            name = entry["attributes"]["name"]
            segmentation = entry.get("segmentation", [])
            view = entry["attributes"]["view"]

            print(f"\n📌 Entry {i}: name='{name}', source='{view}'")

            if not name:
                print("⚠️ Skipping entry with missing name")
                continue

            # Skip if view already exists
            if view and view in mesh_dict[name]["views"]:
                print(f"↪️ Skipping duplicate view '{view}' for '{name}'")
                continue

            # Track unique sources
            if view:
                mesh_dict[name]["views"].add(view)
                mesh_dict[name]["segmentations"][view].extend(segmentation)
                print(f"✅ Added segmentation for '{name}' in view '{view}'")

        except Exception as e:
            print(f"❌ Error processing entry {i}: {e}")

    return mesh_dict


#2.0 Image-to-cartesian coordinates mapping
    # Caller calls 2 callee to handle mesh coordinates conversion from pixel to cartesian 
    # in front, left, back, right orthogonal views.
def callee_map_front_left_segs_from_pixel_2_cartesian(segs,img_width):
    # v = vertical com, h = horizontal com in image coordinates
    if isinstance(segs[0], (list, tuple, np.ndarray)):
        segs = segs[0]

    arr = np.asarray(segs, dtype=np.float32).reshape(-1, 2)

    arr[:, 0] = (2 * arr[:, 0] / img_width) - 1      # normalize h (x)
    arr[:, 1] = 1 - (2 * arr[:, 1] / img_width)     # normalize v (y)

    return arr
def callee_map_back_right_segs_from_pixel_2_cartesian(segs,img_width):
    # v = vertical com, h = horizontal com in image coordinates
    # Flatten in case segs is [[...]]
    if isinstance(segs[0], (list, tuple, np.ndarray)):
        segs = segs[0]

    arr = np.asarray(segs, dtype=np.float32).reshape(-1, 2)

    # Example: back/right might flip x-axis
    arr[:, 0] = (((2 * arr[:, 0] / img_width) - 1))*-1   # flip horizontal
    arr[:, 1] = 1 - (2 * arr[:, 1] / img_width)  # standard vertical mapping

    return arr.tolist()
def caller_preprocess_segmentations(mesh_dict,img_width):
    # Mapping view -> transformation function
    img_width = img_width
    view_map = {
        "front": partial(callee_map_front_left_segs_from_pixel_2_cartesian,img_width =img_width),
        "left": partial(callee_map_front_left_segs_from_pixel_2_cartesian,img_width =img_width),
        "back": partial(callee_map_back_right_segs_from_pixel_2_cartesian,img_width =img_width),
        "right": partial(callee_map_back_right_segs_from_pixel_2_cartesian,img_width =img_width),
    }

    for mesh_name, mesh_data in mesh_dict.items():
        for view, segs in mesh_data["segmentations"].items():
            if view in view_map:
                mesh_data["segmentations"][view] = view_map[view](segs)
            else:
                print(f"⚠️ No mapping function for view '{view}' in mesh '{mesh_name}'")
    return mesh_dict
def export_mesh_dict(mesh_dict, filepath="mesh_dict.json"): ## Export the entire mesh_dict to JSON
    def convert(obj):
        """Recursively convert complex types for JSON serialization."""
        if isinstance(obj, defaultdict):
            obj = dict(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()
        return obj

    clean_dict = convert(mesh_dict)

    with open(filepath, "w") as f:
        json.dump(clean_dict, f, indent=4)

    return filepath
## Visualization For Mapped Mesh Coordinates
def plot_front_view_segmentations(mesh_dict):
    plt.figure(figsize=(6,6))

    all_x, all_y = [], []

    for mesh_name, mesh_data in mesh_dict.items():
        if "front" not in mesh_data["segmentations"]:
            continue

        segs = mesh_data["segmentations"]["front"]

        if isinstance(segs[0], (list, tuple, np.ndarray)) and len(segs) == 1:
            segs = segs[0]

        coords = np.array(segs, dtype=float).reshape(-1, 2)

        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        xs, ys = coords[:,0], coords[:,1]

        all_x.extend(xs)
        all_y.extend(ys)

        plt.plot(xs, ys, marker="o", label=f"{mesh_name}")

    # Autoscale to fit object fully
    plt.xlim(min(all_x)-0.05, max(all_x)+0.05)
    plt.ylim(min(all_y)-0.05, max(all_y)+0.05)

    plt.xlabel("y (Cartesian)")
    plt.ylabel("z (Cartesian)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Front View Segmentations (Autoscaled)")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")  # keep aspect ratio
    plt.show()
def plot_left_view_segmentations(mesh_dict, view="left"):
    plt.figure(figsize=(6,6))

    all_x, all_y = [], []

    for mesh_name, mesh_data in mesh_dict.items():
        if view not in mesh_data["segmentations"]:
            continue

        segs = mesh_data["segmentations"][view]

        # unwrap single nested list
        if isinstance(segs[0], (list, tuple, np.ndarray)) and len(segs) == 1:
            segs = segs[0]

        coords = np.array(segs, dtype=float).reshape(-1, 2)

        
        coords[:, 0] = coords[:, 0]

        xs, ys = coords[:, 0], coords[:, 1]

        # close polygon if not closed
        if not np.array_equal([xs[0], ys[0]], [xs[-1], ys[-1]]):
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])

        all_x.extend(xs)
        all_y.extend(ys)

        plt.plot(xs, ys, marker="o", label=f"{mesh_name}")

    if not all_x or not all_y:
        print(f"No data found for view '{view}'")
        return

    # Autoscale
    plt.xlim(min(all_x)-0.05, max(all_x)+0.05)
    plt.ylim(min(all_y)-0.05, max(all_y)+0.05)

    

    plt.xlabel("x (Cartesian)")
    plt.ylabel("z (Cartesian)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"{view.capitalize()} View Segmentations (autoscaled)")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
def plot_back_view_segmentations(mesh_dict, view="back"):
    plt.figure(figsize=(6,6))

    all_x, all_y = [], []

    for mesh_name, mesh_data in mesh_dict.items():
        if view not in mesh_data["segmentations"]:
            continue

        segs = mesh_data["segmentations"][view]

        # unwrap single nested list
        if isinstance(segs[0], (list, tuple, np.ndarray)) and len(segs) == 1:
            segs = segs[0]

        coords = np.array(segs, dtype=float).reshape(-1, 2)

        # --- Transform for back view ---
        # flip y so that positive y grows left
        coords[:, 0] = coords[:, 0]

        xs, ys = coords[:, 0], coords[:, 1]

        # close polygon if not closed
        if not np.array_equal([xs[0], ys[0]], [xs[-1], ys[-1]]):
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])

        all_x.extend(xs)
        all_y.extend(ys)

        plt.plot(xs, ys, marker="o", label=f"{mesh_name}")

    if not all_x or not all_y:
        print(f"No data found for view '{view}'")
        return

    # Autoscale
    plt.xlim(min(all_x)-0.05, max(all_x)+0.05)
    plt.ylim(min(all_y)-0.05, max(all_y)+0.05)

    # --- Axis directions ---
    # (optional: invert_yaxis if you want z to grow downward)
    plt.gca().invert_xaxis()

    plt.xlabel("y (Cartesian, flipped: grows right→left)")
    plt.ylabel("z (Cartesian)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"{view.capitalize()} View Segmentations (with Flipped Coordinates)")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
def plot_right_view_segmentations(mesh_dict, view="right"):
    plt.figure(figsize=(6,6))

    all_x, all_y = [], []

    for mesh_name, mesh_data in mesh_dict.items():
        if view not in mesh_data["segmentations"]:
            continue

        segs = mesh_data["segmentations"][view]

        # unwrap single nested list
        if isinstance(segs[0], (list, tuple, np.ndarray)) and len(segs) == 1:
            segs = segs[0]

        coords = np.array(segs, dtype=float).reshape(-1, 2)

        # --- Transform for back view ---
        # flip y so that positive y grows left
        coords[:, 0] = coords[:, 0]

        xs, ys = coords[:, 0], coords[:, 1]

        # close polygon if not closed
        if not np.array_equal([xs[0], ys[0]], [xs[-1], ys[-1]]):
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])

        all_x.extend(xs)
        all_y.extend(ys)

        plt.plot(xs, ys, marker="o", label=f"{mesh_name}")

    if not all_x or not all_y:
        print(f"No data found for view '{view}'")
        return

    # Autoscale
    plt.xlim(min(all_x)-0.05, max(all_x)+0.05)
    plt.ylim(min(all_y)-0.05, max(all_y)+0.05)

    # --- Axis directions ---
    # (optional: invert_yaxis if you want z to grow downward)
    plt.gca().invert_xaxis()

    plt.xlabel("x (Cartesian, flipped: grows right→left)")
    plt.ylabel("z (Cartesian)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"{view.capitalize()} View Segmentations (with Flipped Coordinates)")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# 3.0 3D AABB Reconstruction 
    # Compute AABB of each mesh by group all views to get the min x,y,z and max x,y,z
def compute_aabb(mesh_dict):
    """
    Reconstructs AABB for each mesh in mesh_dict.
    Updates the 'aabb' field for each mesh in place.
    """
    for mesh_id, mesh_data in mesh_dict.items():
        views = mesh_data["views"]
        segs = mesh_data["segmentations"]

        x_list, y_list, z_list = [], [], []

        for view in views:
            coords = np.array(segs.get(view, []), dtype=float)
            if coords.size == 0:
                continue

            if view in {"front", "back"}:
                y_list.append(coords[:, 0])  # y from first index
            elif view in {"left", "right"}:
                x_list.append(coords[:, 0])  # x from first index

            z_list.append(coords[:, 1])      # z from second index

        if x_list and y_list and z_list:
            x_all = np.concatenate(x_list)
            y_all = np.concatenate(y_list)
            z_all = np.concatenate(z_list)

            mesh_data["aabb"]["min"] = [float(x_all.min()), float(y_all.min()), float(z_all.min())]
            mesh_data["aabb"]["max"] = [float(x_all.max()), float(y_all.max()), float(z_all.max())]

    return mesh_dict
def export_aabb_only(mesh_dict, filepath="aabb_only.json"): ## Export aabb in form of 3D coordinate pairs
    export_data = {}

    for mesh_id, mesh_data in mesh_dict.items():
        aabb = mesh_data.get("aabb", {})
        # Ensure values are JSON-safe
        min_vals = [float(v) for v in aabb.get("min", [0.0, 0.0, 0.0])]
        max_vals = [float(v) for v in aabb.get("max", [0.0, 0.0, 0.0])]

        export_data[mesh_id] = {
            "aabb": {
                "min": min_vals,
                "max": max_vals
            }
        }

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=4)

    return filepath
## Plot the AABBs (primitive cubes) in 2D orthographic views
def plot_front_view_bouding_boxes(mesh_dict):
    plt.figure(figsize=(6, 6))

    all_x, all_y = [], []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default matplotlib colors

    for idx, (mesh_name, mesh_data) in enumerate(mesh_dict.items()):
        if "front" not in mesh_data["segmentations"]:
            continue

        segs = mesh_data["segmentations"]["front"]

        # Flatten if nested
        if isinstance(segs[0], (list, tuple, np.ndarray)) and len(segs) == 1:
            segs = segs[0]

        coords = np.array(segs, dtype=float).reshape(-1, 2)
        if coords.size == 0:
            continue

        ys, zs = coords[:, 0], coords[:, 1]

        # Bounding box
        y_min, y_max = ys.min(), ys.max()
        z_min, z_max = zs.min(), zs.max()

        all_x.extend([y_min, y_max])
        all_y.extend([z_min, z_max])

        # Draw bounding box in color
        rect_x = [y_min, y_max, y_max, y_min, y_min]
        rect_y = [z_min, z_min, z_max, z_max, z_min]
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(rect_x, rect_y, color=color, linewidth=2, label=f"{mesh_name} bbox")

        # Draw segmentation polygon in black dashed line
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])  # close the loop

        plt.plot(ys, zs, color="black", linestyle="--", linewidth=1)

    # Autoscale
    if all_x and all_y:
        plt.xlim(min(all_x) - 0.05, max(all_x) + 0.05)
        plt.ylim(min(all_y) - 0.05, max(all_y) + 0.05)

    plt.xlabel("y (Cartesian)")
    plt.ylabel("z (Cartesian)")
    plt.title("Front View: Bounding Boxes + Segmentation Masks")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
def plot_left_view_bounding_boxes(mesh_dict):
    plt.figure(figsize=(6, 6))

    all_x, all_y = [], []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default matplotlib colors

    for idx, (mesh_name, mesh_data) in enumerate(mesh_dict.items()):
        if "left" not in mesh_data["segmentations"]:
            continue

        segs = mesh_data["segmentations"]["left"]

        # Flatten if nested
        if isinstance(segs[0], (list, tuple, np.ndarray)) and len(segs) == 1:
            segs = segs[0]

        coords = np.array(segs, dtype=float).reshape(-1, 2)
        if coords.size == 0:
            continue

        xs, zs = coords[:, 0], coords[:, 1]  # left view: x and z

        # Bounding box
        x_min, x_max = xs.min(), xs.max()
        z_min, z_max = zs.min(), zs.max()

        all_x.extend([x_min, x_max])
        all_y.extend([z_min, z_max])

        # Draw bounding box in color
        rect_x = [x_min, x_max, x_max, x_min, x_min]
        rect_y = [z_min, z_min, z_max, z_max, z_min]
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(rect_x, rect_y, color=color, linewidth=2, label=f"{mesh_name} bbox")

        # Draw segmentation polygon in black dashed line
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])  # close the loop

        plt.plot(xs, zs, color="black", linestyle="--", linewidth=1)

    # Autoscale
    if all_x and all_y:
        plt.xlim(min(all_x) - 0.05, max(all_x) + 0.05)
        plt.ylim(min(all_y) - 0.05, max(all_y) + 0.05)

    plt.xlabel("x (Cartesian)")
    plt.ylabel("z (Cartesian)")
    plt.title("Left View: Bounding Boxes + Segmentation Masks")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
def plot_back_view_bounding_boxes(mesh_dict):
    plt.figure(figsize=(6, 6))

    all_x, all_y = [], []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (mesh_name, mesh_data) in enumerate(mesh_dict.items()):
        if "back" not in mesh_data["segmentations"]:
            continue

        segs = mesh_data["segmentations"]["back"]

        # Flatten if nested
        if isinstance(segs[0], (list, tuple, np.ndarray)) and len(segs) == 1:
            segs = segs[0]

        coords = np.array(segs, dtype=float).reshape(-1, 2)
        if coords.size == 0:
            continue

        ys, zs = coords[:, 0], coords[:, 1]  # back view: y and z

        # Bounding box
        y_min, y_max = ys.min(), ys.max()
        z_min, z_max = zs.min(), zs.max()

        all_x.extend([y_min, y_max])
        all_y.extend([z_min, z_max])

        # Draw bounding box in color
        rect_x = [y_min, y_max, y_max, y_min, y_min]
        rect_y = [z_min, z_min, z_max, z_max, z_min]
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(rect_x, rect_y, color=color, linewidth=2, label=f"{mesh_name} bbox")

        # Draw segmentation polygon in black dashed line
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        plt.plot(ys, zs, color="black", linestyle="--", linewidth=1)

    # Autoscale
    if all_x and all_y:
        plt.xlim(min(all_x) - 0.05, max(all_x) + 0.05)
        plt.ylim(min(all_y) - 0.05, max(all_y) + 0.05)

    plt.gca().invert_xaxis()
    plt.xlabel("y (Cartesian)")
    plt.ylabel("z (Cartesian)")
    plt.title("Back View: Bounding Boxes + Segmentation Masks")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
def plot_right_view_bounding_boxes(mesh_dict):
    plt.figure(figsize=(6, 6))

    all_x, all_y = [], []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (mesh_name, mesh_data) in enumerate(mesh_dict.items()):
        if "right" not in mesh_data["segmentations"]:
            continue

        segs = mesh_data["segmentations"]["right"]

        # Flatten if nested
        if isinstance(segs[0], (list, tuple, np.ndarray)) and len(segs) == 1:
            segs = segs[0]

        coords = np.array(segs, dtype=float).reshape(-1, 2)
        if coords.size == 0:
            continue

        coords[:, 0] = coords[:, 0]

        xs, zs = coords[:, 0], coords[:, 1]  # right view: x and z

        # Bounding box
        x_min, x_max = xs.min(), xs.max()
        z_min, z_max = zs.min(), zs.max()

        all_x.extend([x_min, x_max])
        all_y.extend([z_min, z_max])

        # Draw bounding box in color
        rect_x = [x_min, x_max, x_max, x_min, x_min]
        rect_y = [z_min, z_min, z_max, z_max, z_min]
        color = color_cycle[idx % len(color_cycle)]
        plt.plot(rect_x, rect_y, color=color, linewidth=2, label=f"{mesh_name} bbox")

        # Draw segmentation polygon in black dashed line
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        plt.plot(xs, zs, color="black", linestyle="--", linewidth=1)

    # Autoscale
    if all_x and all_y:
        plt.xlim(min(all_x) - 0.05, max(all_x) + 0.05)
        plt.ylim(min(all_y) - 0.05, max(all_y) + 0.05)

    plt.gca().invert_xaxis()
    plt.xlabel("x (Cartesian)")
    plt.ylabel("z (Cartesian)")
    plt.title("Right View: Bounding Boxes + Segmentation Masks")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    json_path_annotation = Path(__file__).resolve().parents[2] / "data" / "json" / "instances_default.json"
    
    #1.0 Annotations Preprossing
    mesh_dict = preprocess_annotations(json_path_annotation) # this function groups the annotations into mesh dictornary
    
    ## print preprocessed annotations
    """""
    print("\n📦 Segmentations for 'body' mesh:")
    print(f"Views: {mesh_dict['body']['views']}")
    for view, segs in mesh_dict["body"]["segmentations"].items():
        print(f"🔹 View '{view}': {len(segs)} segmentation(s)")
    
    print(f"Meshs count: {len(mesh_dict)}")
    print(f"Meshs: {mesh_dict.keys()}")

    print("\n📦 Preprocess_annotations: Segmentations for all meshes:")
    for mesh_name, mesh_data in mesh_dict.items():
        print(f"\n🧩 Mesh: '{mesh_name}'")
        print(f"Views: {mesh_data['views']}")
        for view, segs in mesh_data["segmentations"].items():
            print(f"🔹 View '{view}': {len(segs)} segmentation(s)")
            print(f"  Segmentation: {segs}")
    """""


    # 2.0 Image-to-cartesian coordinates mapping
    img_width =608 # in this case, all images are square, so width = height
    mesh_dict = caller_preprocess_segmentations(mesh_dict,img_width) # this function change the format of segmentations from pixel to cartesian coordinates

    ## print preprocessed segmentations
    '''''
    print("\n📦 Preprocess_segmentations: map view img 2 cart coors:")
    for mesh_name, mesh_data in mesh_dict.items():
        print(f"\n🧩 Mesh: '{mesh_name}'")
        print(f"Views: {mesh_data['views']}")
        for view, segs in mesh_data["segmentations"].items():
            print(f"🔹 View '{view}': {len(segs)} cordinates")
            print(f"  Coordinates: {segs}")
    '''''
    ## draw the segmentations for orthographic views
    plot_front_view_segmentations(mesh_dict)
    plot_left_view_segmentations(mesh_dict, view="left")
    plot_back_view_segmentations(mesh_dict, view="back")
    plot_right_view_segmentations(mesh_dict, view="right")

    print("\n📦 Coordinates for 'right hand' mesh:")    ## print  the coordinates for a specific mesh
    print(f"Views: {mesh_dict['right hand']['views']}")
    for view, segs in mesh_dict["right hand"]["segmentations"].items():
        print(f"🔹 View '{view}': {len(segs)} coordinates")
        print(f"  Coordinates: {segs}")
    
    
    #3.0 3D AABB Reconstruction
    compute_aabb(mesh_dict) # primitive cubes representing AABBs
    print("\n📦 AABB in form of 3D coordinate pairs for all meshes:")    # print the AABB for all meshes
    for mesh_name, mesh_data in mesh_dict.items():
        print(f"🧩 Mesh: '{mesh_name}'")
        print(f"  AABB Min: {mesh_data['aabb']['min']}")
        print(f"  AABB Max: {mesh_data['aabb']['max']}")

    """""
    #export the mesh_dict
    json_path_export_mesh_dict = Path(__file__).resolve().parents[2] / "data" / "json" / "exported_mesh_dict.json"
    export_mesh_dict(mesh_dict, filepath=json_path_export_mesh_dict)
    print(f"\n✅ Exported mesh_dict to: {json_path_export_mesh_dict}")
    """""
    ## Export the AABBs of the mesh_dict
    json_path_export_aabb_only = Path(__file__).resolve().parents[2] / "data" / "json" / "exported_aabb_only.json"
    export_aabb_only(mesh_dict, filepath=json_path_export_aabb_only)
    print(f"\n✅ Exported aabb_only to: {json_path_export_aabb_only}")
    ## draw the bounding boxes for orthographic views
    plot_front_view_bouding_boxes(mesh_dict)
    plot_left_view_bounding_boxes(mesh_dict)
    plot_back_view_bounding_boxes(mesh_dict)
    plot_right_view_bounding_boxes(mesh_dict)


    
