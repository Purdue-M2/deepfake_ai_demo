import streamlit as st
from PIL import Image
import os
import time
import requests
from streamlit_modal import Modal

# Function to load a Lottie animation from a URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Set up the Streamlit page layout and title
st.set_page_config(layout="wide", page_title="Deepfake Demo")
st.title("Preserving Fairness Generalization in Deepfake Detection")

# Image carousel section with right-to-left sliding queue
st.subheader("Explore Deepfake Detection")
carousel_images = [
    {"path": "sample_images/physical11.png", "caption": "Physical Failure"},
    {"path": "sample_images/physical22.png", "caption": "Physical Failure"},
    {"path": "sample_images/physical33.png", "caption": "Physical Failure"},
    {"path": "sample_images/physiological11.png", "caption": "Physiological Failure"},
    {"path": "sample_images/physiological22.png", "caption": "Physiological Failure"},
    {"path": "sample_images/physiological33.jpg", "caption": "Physiological Failure"},
    # {"path": "sample_images/progan.png", "caption": "ProGAN"},
    # {"path": "sample_images/sd15.png", "caption": "Stable Diffusion v1.5"},
    # {"path": "sample_images/stargan.png", "caption": "StarGAN"},
    # {"path": "sample_images/vqgan.jpg", "caption": "VQGAN"},
]  # 10 images for the queue

# Create a directory for sample images if it doesn't exist
sample_dir = "sample_images"
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Generate HTML for the Swiper carousel with right-to-left sliding
swiper_html = """
<link rel="stylesheet" href="https://unpkg.com/swiper/swiper-bundle.min.css" />
<script src="https://unpkg.com/swiper/swiper-bundle.min.js"></script>
<style>
    .swiper-container {
        width: 100%;
        height: 300px;
        overflow: hidden;  /* Hide overflow for a clean look */
    }
    .swiper-slide {
        text-align: center;
        font-size: 18px;
        background: #fff;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .swiper-slide img {
        max-width: 100%;
        max-height: 200px;
        object-fit: cover;
    }
</style>
<div class="swiper-container">
    <div class="swiper-wrapper">
"""

# Add each image to the Swiper carousel
for img in carousel_images:
    if os.path.exists(img["path"]):
        # Convert image to base64 to embed it in HTML
        with open(img["path"], "rb") as f:
            import base64
            img_data = base64.b64encode(f.read()).decode("utf-8")
        swiper_html += f"""
        <div class="swiper-slide">
            <img src="data:image/jpeg;base64,{img_data}" />
            <p>{img['caption']}</p>
        </div>
        """
    else:
        swiper_html += f"""
        <div class="swiper-slide">
            <p>Placeholder: {img['caption']}</p>
        </div>
        """

# Close the Swiper HTML and initialize the carousel with right-to-left sliding
swiper_html += """
    </div>
</div>
<script>
    var swiper = new Swiper('.swiper-container', {
        slidesPerView: 4,  // Show 4 images at a time
        spaceBetween: 10,
        loop: true,  // Loop the images for continuous scrolling
        autoplay: {
            delay: 0,  // No delay for continuous movement
            disableOnInteraction: false,
            reverseDirection: true,  // Move from right to left
        },
        speed: 5000,  // Adjust speed for smooth continuous scrolling (5000ms for the full transition)
    });
</script>
"""

# Render the Swiper carousel
st.components.v1.html(swiper_html, height=350)

# Upload section
st.markdown("---")
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Convert the uploaded image to base64 for embedding in HTML
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Display the uploaded image with limited size using custom HTML and CSS
    st.markdown(
        f"""
        <style>
        .uploaded-image {{
            max-width: 500px;  /* Limit the width */
            max-height: 500px;  /* Limit the height */
            object-fit: contain;  /* Ensure the image scales properly */
            display: block;
            margin: 0 auto;  /* Center the image */
        }}
        .image-caption {{
            text-align: center;
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        </style>
        <div>
            <img src="data:image/png;base64,{img_data}" class="uploaded-image" />
            <div class="image-caption">Your Uploaded Image</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### Processing...")
    # lottie_running = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_tutvdkg0.json")
    # if lottie_running:
    #     st_lottie(lottie_running, height=300, key="running")
    time.sleep(2)
    st.success("Processing Complete!")
    # detection_result = random.choice(["Fake", "Real"])
    detection_result = "Fake"
    st.subheader(f"Detection Result: {detection_result.upper()}")
    sample_dir = "sample_images"
    # if detection_result == "Fake":
    #     result_path = os.path.join(sample_dir, "detection_result_fake.png")
    # else:
    #     result_path = os.path.join(sample_dir, "detection_result_real.png")
    # if os.path.exists(result_path):
    #     result_img = Image.open(result_path)
    #     st.image(result_img, caption=f"Sample Result: {detection_result}", use_container_width=True)
    
    # Pre-convert feature maps, combined feature, and gradcam to base64 for use in the modal
    feature_map1_data = ""
    feature_map2_data = ""
    combined_feature_data = ""
    gradcam_data = ""
    
    feature_map1_path = os.path.join(sample_dir, "feature_map1.png")
    if os.path.exists(feature_map1_path):
        with open(feature_map1_path, "rb") as f:
            feature_map1_data = base64.b64encode(f.read()).decode("utf-8")
    
    feature_map2_path = os.path.join(sample_dir, "feature_map2.png")
    if os.path.exists(feature_map2_path):
        with open(feature_map2_path, "rb") as f:
            feature_map2_data = base64.b64encode(f.read()).decode("utf-8")
    
    combined_feature_path = os.path.join(sample_dir, "combined_feature.png")
    if os.path.exists(combined_feature_path):
        with open(combined_feature_path, "rb") as f:
            combined_feature_data = base64.b64encode(f.read()).decode("utf-8")
    
    gradcam_path = os.path.join(sample_dir, "gradcam.png")
    if os.path.exists(gradcam_path):
        with open(gradcam_path, "rb") as f:
            gradcam_data = base64.b64encode(f.read()).decode("utf-8")
    
    if st.button("Want to know how the model detected it?"):
        modal = Modal("Detection Process Details", key="detection_modal")
        with modal.container():
            st.markdown("### How the Model Detects Deepfakes")
            st.write("Watch the detection process unfold step by step:")
    
            # Create a placeholder for the dynamic process
            process_placeholder = st.empty()
    
            # Step 1: Show the input image
            with process_placeholder.container():
                st.markdown("#### Step 1: Input Image")
                st.markdown(
                    f"""
                    <style>
                    .input-image {{
                        max-width: 300px;  /* Limit the width for the input image in the modal */
                        max-height: 300px;  /* Limit the height */
                        object-fit: contain;  /* Ensure the image scales properly */
                        display: block;
                        margin: 0 auto;  /* Center the image */
                    }}
                    .image-caption {{
                        text-align: center;
                        font-size: 14px;
                        color: #666;
                        margin-top: 5px;
                    }}
                    </style>
                    <div>
                        <img src="data:image/png;base64,{img_data}" class="input-image" />
                        <div class="image-caption">Input Image</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with st.spinner("Extracting features..."):
                time.sleep(2.5)
    
            # Step 2: Show arrows pointing to feature maps
            with process_placeholder.container():
                st.markdown("#### Step 1: Input Image")
                st.markdown(
                    f"""
                    <style>
                    .input-image {{
                        max-width: 300px;
                        max-height: 300px;
                        object-fit: contain;
                        display: block;
                        margin: 0 auto;
                    }}
                    .image-caption {{
                        text-align: center;
                        font-size: 14px;
                        color: #666;
                        margin-top: 5px;
                    }}
                    </style>
                    <div>
                        <img src="data:image/png;base64,{img_data}" class="input-image" />
                        <div class="image-caption">Input Image</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <span style="font-size: 24px;">↓</span>
                        <span style="font-size: 24px; margin-left: 50px;">↓</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("#### Step 2: Feature Extraction")
                st.write("The input image is passed through an encoder to extract two features: forgery and demographic features.")
                cols = st.columns(2)
                with cols[0]:
                    if feature_map1_data:
                        st.markdown(
                            f"""
                            <style>
                            .feature-image {{
                                max-width: 200px;  /* Limit the width for feature maps */
                                max-height: 200px;  /* Limit the height */
                                object-fit: contain;
                                display: block;
                                margin: 0 auto;
                            }}
                            .image-caption {{
                                text-align: center;
                                font-size: 14px;
                                color: #666;
                                margin-top: 5px;
                            }}
                            </style>
                            <div>
                                <img src="data:image/png;base64,{feature_map1_data}" class="feature-image" />
                                <div class="image-caption">Forgery Feature</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("Placeholder: Forgery Feature")
                with cols[1]:
                    if feature_map2_data:
                        st.markdown(
                            f"""
                            <style>
                            .feature-image {{
                                max-width: 200px;
                                max-height: 200px;
                                object-fit: contain;
                                display: block;
                                margin: 0 auto;
                            }}
                            .image-caption {{
                                text-align: center;
                                font-size: 14px;
                                color: #666;
                                margin-top: 5px;
                            }}
                            </style>
                            <div>
                                <img src="data:image/png;base64,{feature_map2_data}" class="feature-image" />
                                <div class="image-caption">Demographic Feature</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("Placeholder: Demographic Feature")
            with st.spinner("Fusing features..."):
                time.sleep(2.5)
    
            # Step 3: Show arrows pointing to combined feature
            with process_placeholder.container():
                st.markdown("#### Step 1: Input Image")
                st.markdown(
                    f"""
                    <style>
                    .input-image {{
                        max-width: 300px;
                        max-height: 300px;
                        object-fit: contain;
                        display: block;
                        margin: 0 auto;
                    }}
                    .image-caption {{
                        text-align: center;
                        font-size: 14px;
                        color: #666;
                        margin-top: 5px;
                    }}
                    </style>
                    <div>
                        <img src="data:image/png;base64,{img_data}" class="input-image" />
                        <div class="image-caption">Input Image</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <span style="font-size: 24px;">↓</span>
                        <span style="font-size: 24px; margin-left: 50px;">↓</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("#### Step 2: Feature Extraction")
                st.write("The input image is passed through an encoder to extract two features: forgery and demographic features.")
                cols = st.columns(2)
                with cols[0]:
                    if feature_map1_data:
                        st.markdown(
                            f"""
                            <style>
                            .feature-image {{
                                max-width: 200px;
                                max-height: 200px;
                                object-fit: contain;
                                display: block;
                                margin: 0 auto;
                            }}
                            .image-caption {{
                                text-align: center;
                                font-size: 14px;
                                color: #666;
                                margin-top: 5px;
                            }}
                            </style>
                            <div>
                                <img src="data:image/png;base64,{feature_map1_data}" class="feature-image" />
                                <div class="image-caption">Forgery Feature</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("Placeholder: Forgery Feature")
                with cols[1]:
                    if feature_map2_data:
                        st.markdown(
                            f"""
                            <style>
                            .feature-image {{
                                max-width: 200px;
                                max-height: 200px;
                                object-fit: contain;
                                display: block;
                                margin: 0 auto;
                            }}
                            .image-caption {{
                                text-align: center;
                                font-size: 14px;
                                color: #666;
                                margin-top: 5px;
                            }}
                            </style>
                            <div>
                                <img src="data:image/png;base64,{feature_map2_data}" class="feature-image" />
                                <div class="image-caption">Demographic Feature</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("Placeholder: Demographic Feature")
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <span style="font-size: 24px;">↓</span>
                        <span style="font-size: 24px; margin-left: 50px;">↓</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("#### Step 3: Feature Fusion")
                st.write("The two features are combined using AdaIN Feature Fusion to create a unified feature representation.")
                if combined_feature_data:
                    st.markdown(
                        f"""
                        <style>
                        .combined-feature-image {{
                            max-width: 300px;  /* Limit the width for the combined feature */
                            max-height: 300px;  /* Limit the height */
                            object-fit: contain;
                            display: block;
                            margin: 0 auto;
                        }}
                        .image-caption {{
                            text-align: center;
                            font-size: 14px;
                            color: #666;
                            margin-top: 5px;
                        }}
                        </style>
                        <div>
                            <img src="data:image/png;base64,{combined_feature_data}" class="combined-feature-image" />
                            <div class="image-caption">Combined Feature</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.write("Placeholder: Combined Feature")
            with st.spinner("Classifying..."):
                time.sleep(2.5)
    
            # Step 4: Show arrow pointing to classification result and Grad-CAM
            with process_placeholder.container():
                st.markdown("#### Step 1: Input Image")
                st.markdown(
                    f"""
                    <style>
                    .input-image {{
                        max-width: 300px;
                        max-height: 300px;
                        object-fit: contain;
                        display: block;
                        margin: 0 auto;
                    }}
                    .image-caption {{
                        text-align: center;
                        font-size: 14px;
                        color: #666;
                        margin-top: 5px;
                    }}
                    </style>
                    <div>
                        <img src="data:image/png;base64,{img_data}" class="input-image" />
                        <div class="image-caption">Input Image</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <span style="font-size: 24px;">↓</span>
                        <span style="font-size: 24px; margin-left: 50px;">↓</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("#### Step 2: Feature Extraction")
                st.write("The input image is passed through an encoder to extract two features: forgery and demographic features.")
                cols = st.columns(2)
                with cols[0]:
                    if feature_map1_data:
                        st.markdown(
                            f"""
                            <style>
                            .feature-image {{
                                max-width rire: 200px;
                                max-height: 200px;
                                object-fit: contain;
                                display: block;
                                margin: 0 auto;
                            }}
                            .image-caption {{
                                text-align: center;
                                font-size: 14px;
                                color: #666;
                                margin-top: 5px;
                            }}
                            </style>
                            <div>
                                <img src="data:image/png;base64,{feature_map1_data}" class="feature-image" />
                                <div class="image-caption">Forgery Feature</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("Placeholder: Forgery Feature")
                with cols[1]:
                    if feature_map2_data:
                        st.markdown(
                            f"""
                            <style>
                            .feature-image {{
                                max-width: 200px;
                                max-height: 200px;
                                object-fit: contain;
                                display: block;
                                margin: 0 auto;
                            }}
                            .image-caption {{
                                text-align: center;
                                font-size: 14px;
                                color: #666;
                                margin-top: 5px;
                            }}
                            </style>
                            <div>
                                <img src="data:image/png;base64,{feature_map2_data}" class="feature-image" />
                                <div class="image-caption">Demographic Feature</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("Placeholder: Demographic Feature")
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <span style="font-size: 24px;">↓</span>
                        <span style="font-size: 24px; margin-left: 50px;">↓</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("#### Step 3: Feature Fusion")
                st.write("The two features are combined using AdaIN Feature Fusion to create a unified feature representation.")
                if combined_feature_data:
                    st.markdown(
                        f"""
                        <style>
                        .combined-feature-image {{
                            max-width: 300px;
                            max-height: 300px;
                            object-fit: contain;
                            display: block;
                            margin: 0 auto;
                        }}
                        .image-caption {{
                            text-align: center;
                            font-size: 14px;
                            color: #666;
                            margin-top: 5px;
                        }}
                        </style>
                        <div>
                            <img src="data:image/png;base64,{combined_feature_data}" class="combined-feature-image" />
                            <div class="image-caption">Combined Feature</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.write("Placeholder: Combined Feature")
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <span style="font-size: 24px;">↓</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("#### Step 4: Classification Head")
                st.write("The combined feature is fed into a classification head to determine if the image is Fake or Real.")
                st.subheader(f"Result: {detection_result.upper()}")
                if gradcam_data:
                    st.markdown(
                        f"""
                        <style>
                        .gradcam-image {{
                            max-width: 300px;  /* Limit the width for the Grad-CAM image */
                            max-height: 300px;  /* Limit the height */
                            object-fit: contain;
                            display: block;
                            margin: 0 auto;
                        }}
                        .image-caption {{
                            text-align: center;
                            font-size: 14px;
                            color: #666;
                            margin-top: 5px;
                        }}
                        </style>
                        <div>
                            <img src="data:image/png;base64,{gradcam_data}" class="gradcam-image" />
                            <div class="image-caption">Grad-CAM Heatmap</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write("The Grad-CAM heatmap highlights the regions of the image that influenced the model's decision.")
                else:
                    st.write("Placeholder: Grad-CAM Heatmap (not available)")
