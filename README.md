
# 🟢 Suggestify

📚 Dataset: https://www.kaggle.com/datasets/krishsharma0413/2-million-songs-from-mpd-with-audio-features
📚 Dataset: https://www.kaggle.com/datasets/abhijithchandradas/rate-your-music-top-albums?resource=download

### 📕 Used libraries:
    - pandas – 2.2.3
    - numpy – 2.1.0
    - Pillow (imported as PIL) – 11.1.0
    - requests – 2.32.3
    - psutil – 7.0.0
    - scikit-learn – 1.6.1
    - spotipy – 2.25.1
    - matplotlib – 3.10.0
    - seaborn – 0.13.2
    - pyspark – 3.5.4
    - scipy – 1.13.1
    - Standard libraries that come with python
        (sys, os, sqlite3, random, time, tkinter, io)

### 🗂️ Contents:
    - data: Folder that contains a link to the used datasets. Since github does not allow large files, they need to be downloaded separately.
    - notebook: Folder containing the Jupyther Notebook in where we have developed our work, with trials, plots, etc.
    - src: Final project to be executed. Consists of:
        - images: Folder containing the images the GUI uses.
        - plots: Folder in where we store the plots when the app finishes.
        - trials: Folder containing other files, equivalent to model.py, we have used for seeng how we could improve the model.
        - GUI.py: Main file of the app. This is what it needs to be executed.
        - model.py: Auxiliary file to GUI.py in where the functions to load the dataset, transform, do PCA, do the recommendations, and others are stored.
        - register.csv: Dataset in where every rating the user has given is stored the following way: [user id, song id, rating(-1,0,1,2)]
    - README.md: This file
    - requirements.txt: Text file mentioning the used libraries and their versions (as seen previously in this file).
    - work-objectives.txt: Text file mentioning some objectives we set in the beginning of the project.

### 💻 How to execute:
    1. Install the necessary libraries (check requirements.txt).
    2. Navigate to the folder 'src'.
    3. Once in the folder 'src', execute GUI.py, making sure it is running inside 'src' and our virtual environment.
