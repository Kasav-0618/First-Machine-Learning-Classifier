# Titanic Classifier Project (VS Code Setup)

This project trains and evaluates a Decision Tree Classifier on the Titanic dataset, compares it against a simple baseline, and computes all required metrics (Accuracy, Confusion Matrix, MSE).

## Files in this Directory
- **`titanic_classification.py`**: Complete, self-contained Python script with terminal output and visualization plots.
- **`titanic_classification.ipynb`**: Interactive Jupyter Notebook ready for VS Code, Google Colab, or GitHub.
- **`submission_writeup.md`**: Ready-to-use project write-up detailing methodology, results, confusion matrix breakdown, and limitations.
- **`requirements.txt`**: Package dependencies.

---

## How to Run in VS Code

### Option 1: Run as a Python Script (`.py`)
1. Open this folder in VS Code:
   ```bash
   code C:\Users\lsk22\.gemini\antigravity\scratch\titanic_classifier
   ```
2. Open the terminal inside VS Code (`Ctrl + ` `~``) and install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python titanic_classification.py
   ```
4. Three visualization images will be saved automatically in the folder:
   - `confusion_matrix.png`
   - `decision_tree.png`
   - `model_comparison.png`

---

### Option 2: Run as a Jupyter Notebook (`.ipynb`)
1. In VS Code, install the **Jupyter** extension (by Microsoft) from the Extensions view (`Ctrl + Shift + X`).
2. Open `titanic_classification.ipynb`.
3. Select your Python kernel in the top-right corner.
4. Click **Run All** to run all cells interactively.

---

## How to Submit Your Project Link
Per assignment guidelines:
1. **GitHub Option:**
   - Push this folder to a public GitHub repository.
   - Copy the repository URL (or direct file URL to `titanic_classification.ipynb`).
2. **Google Drive / Google Colab Option:**
   - Upload `titanic_classification.ipynb` to your Google Drive.
   - Right-click -> **Open with Google Colaboratory**.
   - Click **Share** (top right) -> change General access to **"Anyone with the link can view"**.
   - Copy the shareable link and paste it into your assignment submission along with `submission_writeup.md`.
