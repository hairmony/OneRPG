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

1. Download dependencies by doing py -m pip install -r Dependencies.txt
2. Run `UserProfileArchetype.py` to prepare the model
3. Run `BuildClusters.py` to create a file called `cluster_profiles.json`
4. Open OneRPG.py on command line
5. Launch game by entering the following in your terminal:
    ```bash
    python -m streamlit run "OneRPG.py"
    ```
MIT License
Copyright (c) 2025
Nicholas Roy, Hariharan Vallath.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
