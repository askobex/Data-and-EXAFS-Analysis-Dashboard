# BS.py: Data Reduction & EXAFS Analysis Dashboard

**BS.py** is a high-performance Python library designed for the Canadian Light Source (CLS) BioXAS beamlines. It streamlines the transition from raw beamline data to publication-ready EXAFS analysis through an integrated suite of processing tools and interactive dashboards.

---

## 🚀 Key Features

### 📂 Intelligent Path Handling
* **Beamline Integration:** Automatically resolves complex directory structures for `BIOXAS-MAIN` and `BIOXAS-SPECTROSCOPY`.
* **Remote Access:** Built-in support for mapping project IDs to network drives (e.g., `Z:\projects`).

### 🛠️ Advanced Data Cleaning
* **XASDeglitcher:** A robust class for noise removal using local statistics (Median and Median Absolute Deviation).
* **Region-Specific Processing:** Targeted cleaning for Pre-edge, XANES, or EXAFS regions to preserve data integrity while removing spikes.

### 🧪 Core EXAFS Analysis (Powered by Larch)
* **Normalization:** Automated E0 determination and pre-edge/post-edge flattening.
* **Background Subtraction:** Fine-tuned `autobk` implementation for extracting $\chi(k)$.
* **Signal Transformation:** High-fidelity Fourier Transforms for moving between Energy, $k$-space, and $R$-space.

### 📊 Interactive Dashboards
* **Jupyter Integration:** Real-time data visualization using `ipywidgets`.
* **Dynamic Tuning:** Adjust normalization and background parameters on the fly with immediate visual feedback.

---

## 🛠 Installation

Install the required dependencies using the provided `Requirement.txt`:

```bash
pip install -r Requirement.txt
```

**Main Requirements:**
* `larch`: Core XAFS processing algorithms.
* `ipywidgets`: User interface and dashboard elements.
* `numpy` / `scipy`: Numerical processing and interpolation.
* `matplotlib`: Data visualization.

---

## 📖 Quick Start

1.  **Set Environment:** Open `Testing.ipynb` in a Jupyter environment.
2.  **Initialize Project:**
    ```python
    from BS import project_number
    data_path = project_number(project_id="1234", session=None)
    ```
3.  **Launch Dashboard:** Execute the dashboard cells to start selecting files, deglitching, and performing EXAFS analysis.

---

## 📂 Repository Structure

* **`BS.py`**: The main library containing the `XASDeglitcher` and dashboard classes.
* **`Testing.ipynb`**: A ready-to-use template for your analysis workflow.
* **`User Manual.pdf`**: Comprehensive guide on theory and software usage.
* **`Requirement.txt`**: List of Python dependencies.

---
*Developed for advanced X-ray Absorption Spectroscopy workflows at the BioXAS beamlines.*
