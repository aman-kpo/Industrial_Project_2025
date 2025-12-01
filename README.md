# Peak detection Tkinter Application

### This is a Tkinter Based Application which is used to determine the number of tumor(peaks) in a signal data.
Here's a quick guide to use this in your local system.
1. Clone this repository using by typing the below code:
   ```
   git clone https://github.com/aman-kpo/Industrial_Project_2025.git
   ```
2. Now install the required depencies to run the app using the following command:
   ```
   pip install -r requirements.txt
   ```
3. Now run the main.py file using the following command:
   ```
   python main.py
   ```

4. Now in the default you want to import the model_peak_detect.py in the mport model option
5. Then upload the data you want to visualize the peak for.



---



## Peak Detection by Moving Averages Cascade (Notebook)
*Added on 2025-12-01*

This section documents the `Peak_Detection_by_Moving_Averages_Cascade_2025_1201.ipynb` notebook. The code implements a robust signal processing pipeline for stereo channel data analysis.

### Key Features
* **Universal Data Loading:** Works without Google Drive dependencies.
* **Dynamic Configuration:** Easily switch between **2000 Hz** and **5000 Hz** data sources.
* **Auto-Packaging:** Automatically zips results (logs, dumps, plots).

### How to Run
1.  Open the `.ipynb` file in Google Colab.
2.  **Input Data:** Upload your input CSV file (e.g., the 5000Hz dataset or your own 2000Hz file).
3.  **Configuration:** In the notebook's **Configuration** form, ensure `INPUT_FILENAME` and `SAMPLE_RATE_HZ` match.
4.  Run all cells.

### Output
Generates metadata logs (`detection_log`), raw data segments (`segments_dump`), and visualization plots.



---


