# AI Model Packaging Library

## Overview

This repository contains the **AI Model Packaging Library**, developed as part of the **Capstone Project (Milestone 5) for the Master of Science in Software Engineering** at **Grand Canyon University (GCU)**.
The project delivers a **command-line interface (CLI) tool** that automates the process of packaging machine learning models into deployable **Docker containers**—eliminating the need for manual containerization and DevOps expertise.

---

## Project Purpose

The AI Model Packaging Library addresses a **common challenge in machine learning workflows**: deploying trained models into production.
Traditionally, this requires:

* Manual creation of Dockerfiles
* Dependency management for different frameworks
* Model and inference script integration
* Knowledge of Docker, containerization, and DevOps pipelines

This tool solves those problems by offering a **single command** that packages any supported ML model into a fully deployable Docker image.

---

## Key Features

* **Model Auto-Detection**: Supports **PyTorch (.pth)** and **TensorFlow (.h5)** formats.
* **Automated Dockerization**: Generates Dockerfile, installs dependencies, and packages the model.
* **Zero Docker Expertise Required**: Ideal for data scientists unfamiliar with containerization.
* **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux with Docker installed.
* **Production-Ready Images**: Includes inference script and sample input for testing.
* **Error Handling & Fallbacks**: Gracefully handles build failures and provides troubleshooting guidance.

---

## Repository Structure

```
capstone5/
│
├── cli.py                  # Main CLI entry point
├── docker_packager.py      # Docker build logic
├── infer.py                # Inference script inside container
├── models/
│   └── gen_real_model.py   # Script to generate a real pretrained model
├── requirements.txt        # Python dependencies
├── sample.jpg              # Example input image for inference
└── resnet18_full.pth       # Generated model file (not included in repo by default)
```

---

## 🛠 Installation & Setup

### **1. Clone Repository**

```bash
git clone https://github.com/yourusername/ai-model-packaging-library.git
cd ai-model-packaging-library
```

### **2. Install Dependencies**

Make sure you have Python 3.9+ and Docker installed.

```bash
pip install -r requirements.txt
```

### **3. Generate a Sample Model**

```bash
python models/gen_real_model.py
```

### **4. Package the Model into a Docker Image**

```bash
python cli.py --input resnet18_full.pth --image my_ai_model:1.0
```

### **5. Run Inference Inside the Container**

```bash
docker run --rm my_ai_model:1.0
```

---

## 📊 Example Output

**Building Docker Image**

```
SUCCESS: Built Docker image: my_ai_model:1.0
```

**Running Inference**

```
Processing image: sample.jpg
Top 5 predictions:
   1. Class 741: 7.2%
   2. Class 539: 5.1%
   3. Class 735: 4.3%
```

---

## Security Considerations

* Uses **end-to-end Docker container isolation**.
* All dependencies installed from trusted package sources.
* Encourages use of **private container registries** for sensitive models.

---

## References

Amazon Web Services. (2024). *Amazon Elastic Container Service documentation*. [https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html)
Docker, Inc. (2024). *Docker overview*. [https://docs.docker.com/get-started/overview/](https://docs.docker.com/get-started/overview/)
Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.
Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*. [https://doi.org/10.48550/arXiv.1912.01703](https://doi.org/10.48550/arXiv.1912.01703)
Red Hat. (2023). *Introduction to containers, Kubernetes, and Red Hat OpenShift*. [https://www.redhat.com/en/topics/containers](https://www.redhat.com/en/topics/containers)

---

## Academic Information

This project was developed as part of **SWE-550: Master’s Capstone in Software Engineering** at **Grand Canyon University**. It demonstrates **software engineering, DevOps integration, and applied machine learning deployment skills**.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
