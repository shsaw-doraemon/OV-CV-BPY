# Description: This is a module for uploading multi-view images in a Streamlit app.

# This module let user upload 4 orthogonal images (front, left, right, back) and resize it
# Save the images in ./data/input/uploaded_image/ folder
# The resized images can be downloaded, and user can manually segments the meshes in
# each view using cvat tool (Segment Anything Model is used for semi-automatic segmentation)
# Export the annotations in COCO 1.0 format (in CVAT)



import streamlit as st
from PIL import Image
from pathlib import Path

def upload_image():
    st.set_page_config(page_title="Multi-View Image Upload", layout="centered")

    # Initialize session state
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    # If submitted, show dummy page
    if st.session_state.submitted:
        st.title("🎉 You win!")
        st.success("Integration test passed. This is your dummy success page.")
        return True

    else:
        st.title("📸 Upload Multi-View Images")

        views = {
            "View 1: Front": "front",
            "View 2: Left": "left",
            "View 3: Right": "right",
            "View 4: Back": "back"
        }

        uploaded_images = {key: None for key in views.values()}
        raw_files = {key: None for key in views.values()}

        # Front view (required)
        st.subheader("View 1: Front")
        front_file = st.file_uploader("Upload front view image (required)", type=["jpg", "jpeg", "png"], key="front")

        if front_file:
            img = Image.open(front_file)
            img = img.resize((608, 608))  
            uploaded_images["front"] = img
            raw_files["front"] = img
            st.image(uploaded_images["front"], caption="Front View Preview", width=250)
            st.success("Front view image loaded and resized.")

            # Optional views
            for label, key in list(views.items())[1:]:
                st.subheader(label)
                file = st.file_uploader(f"Upload {key} view image", type=["jpg", "jpeg", "png"], key=key)
                if file:
                    img = Image.open(file)
                    img = img.resize((1024, 1024))  # Resize to 1024x1024
                    uploaded_images[key] = img
                    raw_files[key] = img
                    st.image(uploaded_images[key], caption=f"{label} Preview", width=250)
                    st.success(f"{label} image loaded and resized.")
                else:
                    st.info(f"Optional: Upload the {key} view image.")

            # Submit button
            if st.button("✅ Submit"):
                # Define target directory using pathlib
                save_dir = Path("data") / "input" / "uploaded_image"
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save each image with fixed naming
                for key in ["front", "left", "right", "back"]:
                    if raw_files[key]:
                        save_path = save_dir / f"{key}.png"
                        raw_files[key].save(save_path, format="PNG")
                        st.success(f"Saved {key} view as {save_path.name}")

                st.session_state.submitted = True
                st.rerun()

        else:
            st.warning("⚠️ You must upload a front view image before proceeding.")

if __name__ == "__main__":
    upload_image()
    


