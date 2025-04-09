import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import psycopg2
from datetime import datetime
import os
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# Define the CNN model (same as in training)
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()  # Updated super() call for Python 3.12
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    
    # Add error handling for model loading
    try:
        model_state = torch.load('mnist_model.pth', map_location=device)
        model.load_state_dict(model_state)
        print("Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
    
    model.eval()
    return model, device

# Function to preprocess drawn image
def preprocess_image(image):
    # Convert to grayscale and resize to 28x28
    image = image.convert('L')
    image = image.resize((28, 28))
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(image)

# Function to get prediction with confidence
def get_prediction_with_confidence(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class and its probability
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities.cpu().numpy()[0]

# Function to connect to PostgreSQL database
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST', 'db'),
            database=os.environ.get('DB_NAME', 'mnist_app'),
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', 'postgres')
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        print(f"Database connection error: {e}")
        return None

# Function to log prediction to database
def log_prediction(predicted_digit, true_label, confidence):
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Insert prediction data
        cursor.execute(
            "INSERT INTO predictions (timestamp, predicted_digit, true_label, confidence) VALUES (%s, %s, %s, %s)",
            (datetime.now(), predicted_digit, true_label, confidence)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

# Function to fetch prediction data from database
def get_prediction_data(limit=50):
    try:
        conn = get_db_connection()
        if not conn:
            return None
            
        cursor = conn.cursor()
        
        # Get prediction data with newest first
        cursor.execute(
            "SELECT timestamp, predicted_digit, true_label, confidence FROM predictions ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        
        # Fetch all rows and column names
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        cursor.close()
        conn.close()
        
        # Create dataframe
        df = pd.DataFrame(rows, columns=columns)
        
        # Format the confidence as percentage
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2f}")
        
        # Format timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    except Exception as e:
        st.error(f"Error fetching prediction data: {e}")
        return None

# Callback function for digit selection
def select_digit(digit):
    st.session_state.true_label = digit

# Streamlit app
def main():
    st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✏️")
    
    # Initialize session state variables
    if 'true_label' not in st.session_state:
        st.session_state.true_label = None
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = False
    
    st.title("Handwritten Digit Recognition")
    
    # Load model
    model, device = load_model()
    
    # Set up canvas for drawing
    col1, col2 = st.columns([3, 2])
    
    with col1:
        canvas_result = st_canvas(
            fill_color="green",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        predict_button = st.button("Predict")
        st.markdown("### Instructions:")
        st.markdown("1. Draw a digit (0-9) in the canvas and click 'Predict' to see the result")
        st.markdown("2. Select the correct digit and click 'Submit True Label' to save")
        st.markdown("3. Click the Trash icon and repeat")
    
    with col2:
        # If predict button is clicked and there's something drawn on the canvas
        if predict_button and canvas_result.image_data is not None:
            try:
                # Convert canvas image to PIL Image
                image_data = canvas_result.image_data
                image = Image.fromarray(image_data.astype('uint8'))
                
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Get prediction and confidence
                predicted_digit, confidence, all_probs = get_prediction_with_confidence(model, processed_image, device)
                
                # Store the prediction data in session state
                st.session_state.prediction_data = {
                    'predicted_digit': predicted_digit,
                    'confidence': confidence,
                    'probabilities': all_probs
                }
                st.session_state.image_processed = True
                st.session_state.true_label = None
                
                # Force a rerun to ensure we're using the updated session state
                st.rerun()
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.exception(e)  # Show detailed exception for debugging
        
        # Display prediction results if available
        if st.session_state.image_processed and st.session_state.prediction_data:
            data = st.session_state.prediction_data
            predicted_digit = data['predicted_digit']
            confidence = data['confidence']
            all_probs = data['probabilities']
            
            st.markdown(f"**Prediction**: {predicted_digit}")
            st.markdown(f"**Confidence**: {confidence:.2f}")
            st.markdown("**All digit probabilities**:")
            prob_df = {"Digit": list(range(10)), "Probability": all_probs}
            st.bar_chart(prob_df, x="Digit", y="Probability", horizontal=True)
            
            st.markdown("### Select the true digit:")
            digit_container = st.container()

            cols = digit_container.columns(10)

            # Add a button for each digit in its own column
            for i in range(10):
                with cols[i]:
                    # Check if this is the currently selected digit
                    is_selected = st.session_state.true_label == i
                    
                    # Use 'primary' button type when selected
                    button_type = "primary" if is_selected else "secondary"
                    
                    st.button(
                        f"{i}", 
                        key=f"digit_{i}",
                        on_click=select_digit,
                        args=(i,),
                        type=button_type  # This changes the visual appearance based on selection
                    )
            
            # Button to log prediction to database
            if st.button("Submit True Label"):
                if st.session_state.true_label is None:
                    st.warning("Please select a true digit before submitting.")
                elif log_prediction(predicted_digit, st.session_state.true_label, confidence):
                    st.success("Prediction logged successfully!")
                    # Keep the prediction but reset the selection
                    st.session_state.true_label = None
                else:
                    st.error("Failed to log prediction.")
    
    # Display database contents at the bottom of the page
    st.markdown("---")
    st.header("Database Contents")
    
    # Add a refresh button
    if st.button("Refresh Database View"):
        st.rerun()
    
    # Fetch and display prediction data
    prediction_data = get_prediction_data()
    if prediction_data is not None and not prediction_data.empty:
        st.dataframe(
            prediction_data,
            column_config={
                "timestamp": "Timestamp",
                "predicted_digit": "Predicted Digit",
                "true_label": "True Digit",
                "confidence": "Confidence"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No prediction data available. Try making some predictions first!")

if __name__ == "__main__":
    main()