import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import extractNumberPlates, recogFunc


def main():
    st.title("Number Plate Recognition")
    st.caption("Try uploading an image")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        # Convert the image to NumPy array
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        number_plates = extractNumberPlates(image)
        if number_plates:
            for i, plate in enumerate(number_plates):
                plate_val, _ = recogFunc(plate)
                
                if len(plate_val) > 0:
                    st.divider()
                    st.header(f"Plate {i+1}")
                    st.image(plate)
                    st.write(f"Plate Number: {plate_val}")
        else:
            st.write("No number plates detected! Use another image..")

if __name__ == "__main__":
    main()