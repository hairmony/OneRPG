# OneRPG

**OneRPG** is a strategic, Uno-based card game built on Streamlib. It transforms the basic Uno system into a role-playing game with classes, abilities, and stats. Players will get assigned a class using LLM-driven profiling based on their entered Reddit comments. The project uses K-means clustering to generate unique stats for each player using the r/Canada database.

## Requirements

* Python 3.9+
* `streamlit`
* `numpy`
* `scikit-learn`
* `joblib`
* `requests`

## Running the App

1. Run `UserProfileArchetype.py` to prepare the model
2. Run `BuildClusters.py` to create a file called `cluter_profiles.json`
3. Open OneRPG.py on command line
4. Launch game by entering the following in your terminal:
    ```bash
    python -m streamlit run "OneRPG.py"
    ```
