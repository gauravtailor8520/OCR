import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from transformers import AutoModel

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C://Users//91789//Documents//Desktop//Parimal//ocr_env//tesseract.exe'  # This path is for Colab

# Load the model (if needed, you can remove this if not using the model)
model = AutoModel.from_pretrained("stepfun-ai/GOT-OCR2_0", trust_remote_code=True)

def extract_text_from_image(image):
    # Convert the image to a format suitable for OpenCV
    image = np.array(image)

    # Convert the image to RGB (from BGR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use pytesseract to do OCR on the image for both Hindi and English
    extracted_text = pytesseract.image_to_string(rgb_image, lang='hin+eng')

    return extracted_text

def main():
    st.title("Image OCR with Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Extract text from the uploaded image
        extracted_text = extract_text_from_image(image)

        # Display the extracted text
        st.write("Extracted Text:")
        st.write(extracted_text)

if __name__ == "__main__":
    main()
