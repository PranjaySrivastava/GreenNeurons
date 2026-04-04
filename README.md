# GreenNeurons

# PM2.5 Forecasting using Spatio-Temporal Deep Learning

## Overview

This project focuses on predicting PM2.5 (air pollution) levels using deep learning. Since pollution depends on both time (how it changes) and space (how it spreads), we treat it as a spatio-temporal problem.

The goal is to use past data (weather + emissions) to predict future pollution levels accurately.

## Approach

We use a ConvLSTM-based model to capture both spatial and temporal patterns.

Some key ideas we used:

- **ConvLSTM** to learn how pollution evolves over time and across locations  
- **Residual learning**, where the model predicts change instead of absolute values  
- **Multiple input features** like temperature, wind, pressure, rain, etc.  
- **Peak-aware loss** to better handle high pollution events  
- **Post-processing (persistence blending)** to make predictions more realistic  


## Model Pipeline


Input (10 timesteps)
↓
Feature Fusion
↓
ConvLSTM Layers
↓
Residual Prediction (Δ PM2.5)
↓
Forecast (16 timesteps)
↓
Post-processing
↓
Final Output


---

## Results

- Baseline: ~0.85  
- Improved model: ~0.90–0.92  

We observed better stability and improved predictions during high pollution periods.

---

## How to Run

**Requirements:**
- Python  
- PyTorch  
- NumPy  

**Steps:**
1. Place dataset in `/kaggle/input/...`
2. Run the training script  
3. Output will be saved as:


/kaggle/working/preds.npy


---

## Limitations

- Performance depends on data quality  
- Extreme pollution spikes are still hard to predict  
- No explicit physical modeling (purely data-driven)

---

## Future Work

- Try model ensembling  
- Add physics-based constraints  
- Explore transformer models  

---

## License

This project follows the **ANRF Open License**.

You are free to use, modify, and share the work, as long as proper credit is given.  
However, it should not be used for harmful, unethical, or illegal purposes.

---

## Acknowledgements

- ANRF AISE Hackathon  
- Open-source community  

---

## Contact

Feel free to reach out for any questions or collaboration.
