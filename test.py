from detectron2.engine import DefaultPredictor

import os
import pickle
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO


from utils import *

cfg_save_path = ".\detectron2\OD_BC_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

predictor = DefaultPredictor(cfg)

if __name__ == '__main__':
    st.header('BARCOUNT', divider='rainbow')
    st.header('Model used: :blue[detectron2]')
    uploaded_img = st.file_uploader("Upload image", type = ['jpeg', 'png', 'jpg'])
    if uploaded_img is not None:
            st.write("Image uploaded successfully!")
            st.empty()
            st.markdown("---")
            if st.button("Detect bars and give the count"):
                with st.spinner("... This may take a while ⏳"):
                     on_image(uploaded_img, predictor)