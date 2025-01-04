import streamlit as st
import cv2
import torch
import os
import tempfile
from video_utils import (
    predict_video_sequences, 
    display_video_sequences, 
    generate_videos, 
    generate_actual_video
)
from Models import ConvLSTMModel, PredRNN, VideoTransformer
from Models import Config

# Function to initialize the ConvLSTM model
def prepare_convlstm_model():
    model = ConvLSTMModel(
        input_channels=3,
        hidden_channels=[128, 64, 64],
        kernel_size=(3, 3),
        num_layers=3,
        output_channels=3,
        output_frames=5
    )
    model.load_state_dict(torch.load('weights/convlstm_model.pth', map_location=torch.device('cpu')))
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model

# Function to initialize the PredRNN model
def prepare_predrnn_model():
    config = Config()
    model = PredRNN(config).to(config.device)
    model.load_state_dict(torch.load('weights/predrnn_model.pth', map_location=torch.device('cpu')))
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model

# Function to initialize the PredRNN model
def prepare_transformer_model():
    model = VideoTransformer(input_frames=10, output_frames=5, frame_size=(64, 64), color_channels=3).to(torch.device('cpu'))
    # Load the weights into the model
    checkpoint = torch.load('weights/video_transformer_model.pth', map_location=torch.device('cpu'))
    # Load the model weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model

# Function to read video file as bytes
def read_video_bytes(video_path):
    with open(video_path, "rb") as video_file:
        return video_file.read()

# Function to log video properties
def log_video_properties(video_path, label):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        st.info(f"{label} - Width: {width}, Height: {height}, FPS: {fps}, Frames: {frame_count}")
        cap.release()
    else:
        st.error(f"❌ Cannot open {label} for properties.")

# Streamlit UI setup
st.set_page_config(
    page_title="📹 Video Frame Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📹 Frame Prediction in Video Streams")

# Sidebar for Inputs
with st.sidebar:
    st.header("🔧 Settings")
    
    # Upload video file
    uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
    
    # Model selection
    model_option = st.selectbox("Select Model for Prediction", ("ConvLSTM", "PredRNN", "Transformer"))
    
    # Sequence length slider
    sequence_length = st.slider("Sequence Length for Prediction", min_value=1, max_value=10, value=5)
    
    # Sharpening toggle
    sharpen = st.checkbox("Apply Sharpening to Predictions", value=False)
    
    # Submit button
    process_button = st.button("🛠️ Process Video")

# Main Area for Outputs
if uploaded_video is not None and process_button:
    with st.spinner("🔄 Processing the uploaded video..."):
        try:
            # Set the output directory path
            output_dir = os.path.abspath(os.path.join(os.getcwd(), "output_videos"))
            os.makedirs(output_dir, exist_ok=True)
            
            output_video_predicted = os.path.join(output_dir, "predicted_output.mp4")
            output_video_actual = os.path.join(output_dir, "actual_output.mp4")
            
            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_video.read())
                video_path = temp_video.name
            
            st.success(f"✅ Uploaded video saved to {video_path}")
            
            # Extract FPS from the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"❌ Error opening video file {video_path}")
                st.stop()
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                st.success(f"✅ Uploaded video FPS: {fps:.2f}")
                cap.release()
            
            # Load the selected model
            st.info(f"📥 Loading {model_option} model...")
            if model_option == "ConvLSTM":
                model = prepare_convlstm_model()
            elif model_option == "PredRNN":
                model = prepare_predrnn_model()
            elif model_option == "Transformer":
                model = None # prepare_transformer_model()
            else:
                st.error("❌ Invalid model selection.")
                st.stop()
            
            # Predict video sequences
            st.info("🧠 Predicting video sequences...")
            input_sequences, predicted_sequences, actual_sequences, metrics_sequences = predict_video_sequences(
                model=model,
                video_path=video_path,
                num_sequences=sequence_length,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                input_frames=10,
                output_frames=5,
                frame_size=(64, 64)
            )
            
            # Display video sequences
            st.info("📊 Displaying video sequences...")
            display_video_sequences(
                input_sequences=input_sequences,
                predicted_sequences=predicted_sequences,
                metrics_sequences=metrics_sequences,
                display_size=(320, 240),
                is_sharpening=sharpen
            )
            
            # Generate predicted video
            st.info("🎥 Generating predicted video...")
            generate_videos(
                model=model,
                video_path=video_path,
                output_path_predicted=output_video_predicted,
                frame_size=(320, 240),  # Output video size (width, height)
                frame_rate=int(fps),    # Pass FPS extracted from the uploaded video
                input_frames=10,
                output_frames=5
            )
            
            # Generate actual video
            st.info("🎞️ Generating actual video...")
            generate_actual_video(
                video_path=video_path,
                output_path_actual=output_video_actual,
                frame_size=(320, 240),  # Output video size (width, height)
                frame_rate=int(fps)     # Use the same FPS as the uploaded video
            )
            
            # Verify video generation
            st.markdown("### 📹 Generated Videos")
            if os.path.exists(output_video_predicted):
                st.success(f"✅ Predicted video generated at {output_video_predicted} (Size: {os.path.getsize(output_video_predicted)} bytes)")
                log_video_properties(output_video_predicted, "Predicted Video")
            else:
                st.error("❌ Failed to generate predicted video.")
            
            if os.path.exists(output_video_actual):
                st.success(f"✅ Actual video generated at {output_video_actual} (Size: {os.path.getsize(output_video_actual)} bytes)")
                log_video_properties(output_video_actual, "Actual Video")
            else:
                st.error("❌ Failed to generate actual video.")
            
            # Display both videos side by side using Streamlit's built-in video player with byte streams
            st.markdown("---")
            st.subheader("📺 Predicted vs Actual Videos")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Predicted Video")
                if os.path.exists(output_video_predicted):
                    video_pred_bytes = read_video_bytes(output_video_predicted)
                    st.video(video_pred_bytes, format="video/mp4")
                    # Provide download button
                    st.download_button(
                        label="📥 Download Predicted Video",
                        data=video_pred_bytes,
                        file_name="predicted_output.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("❌ Predicted video not found.")
            
            with col2:
                st.markdown("### 🎬 Actual Video")
                if os.path.exists(output_video_actual):
                    video_act_bytes = read_video_bytes(output_video_actual)
                    st.video(video_act_bytes, format="video/mp4")
                    # Provide download button
                    st.download_button(
                        label="📥 Download Actual Video",
                        data=video_act_bytes,
                        file_name="actual_output.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("❌ Actual video not found.")
            
            st.success("🎉 Video processing and display completed!")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
        
        finally:
            # Cleanup temporary video file
            if os.path.exists(video_path):
                os.remove(video_path)
                st.warning("🗑️ Temporary video file removed.")
            
            # Optionally, remove generated videos after displaying
            # if os.path.exists(output_video_predicted):
            #     os.remove(output_video_predicted)
            # if os.path.exists(output_video_actual):
            #     os.remove(output_video_actual)

elif uploaded_video is not None and not process_button:
    st.info("📝 Click the '🛠️ Process Video' button to start processing.")
