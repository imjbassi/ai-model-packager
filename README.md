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
ai-model-packager/
│
├── cli.py                  # Main CLI entry point
├── docker_packager.py      # Docker build logic
├── infer.py                # Inference script inside container
├── models/
│   └── gen_real_model.py   # Script to generate a real pretrained model
├── requirements.txt        # Python dependencies
├── sample.jpg              # Example input image for inference
├── resnet18_full.pth       # Generated model file (not included in repo by default)
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## Installation & Setup

### **Prerequisites**

* Python 3.9 or higher
* Docker Desktop or Docker Engine installed and running
* Git (for cloning the repository)

### **1. Clone Repository**

```bash
git clone https://github.com/imjbassi/ai-model-packager.git
cd ai-model-packager
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Generate a Sample Model**

```bash
python models/gen_real_model.py
```

This creates a `resnet18_full.pth` file in the project root directory.

### **4. Package the Model into a Docker Image**

```bash
python cli.py --input resnet18_full.pth --image my_ai_model:1.0
```

### **5. Run Inference Inside the Container**

```bash
docker run --rm my_ai_model:1.0
```

---

## Usage

### **Basic Command**

```bash
python cli.py --input <model_file> --image <image_name:tag>
```

### **Arguments**

* `--input`: Path to the model file (`.pth` for PyTorch or `.h5` for TensorFlow)
* `--image`: Name and tag for the Docker image (e.g., `my_model:1.0`)

### **Example**

```bash
python cli.py --input resnet18_full.pth --image resnet_inference:latest
```

---

## Example Output

### **Building Docker Image**

```
Building Docker image...
SUCCESS: Built Docker image: my_ai_model:1.0
```

### **Running Inference**

```
Processing image: sample.jpg
Top 5 predictions:
   1. Class 741: 7.2%
   2. Class 539: 5.1%
   3. Class 735: 4.3%
   4. Class 604: 3.8%
   5. Class 892: 2.9%
```

---

## Security Considerations

* Uses **end-to-end Docker container isolation** to prevent unauthorized access.
* All dependencies are installed from **trusted package sources** (PyPI).
* Encourages use of **private container registries** for sensitive or proprietary models.
* **Recommendation**: Scan Docker images for vulnerabilities using tools like Docker Scout or Trivy before deployment.

---

## Troubleshooting

### **Docker Build Fails**

* Ensure Docker is running: `docker ps`
* Check Docker daemon logs for errors
* Verify sufficient disk space for image layers

### **Model File Not Found**

* Confirm the model file path is correct
* Ensure the model was generated successfully using `gen_real_model.py`

### **Permission Errors**

* On Linux/macOS, you may need to run Docker commands with `sudo` or add your user to the `docker` group

---

## Future Enhancements

* Support for additional frameworks (ONNX, scikit-learn, XGBoost)
* Integration with cloud deployment platforms (AWS, Azure, GCP)
* Automated model versioning and registry management
* REST API wrapper generation for containerized models

---

## References

Amazon Web Services. (2024). *Amazon Elastic Container Service documentation*. https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html

Docker, Inc. (2024). *Docker overview*. https://docs.docker.com/get-started/overview/

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*. https://doi.org/10.48550/arXiv.1912.01703

Red Hat. (2023). *Introduction to containers, Kubernetes, and Red Hat OpenShift*. https://www.redhat.com/en/topics/containers

---

## Academic Information

This project was developed as part of the **Master's Capstone in Software Engineering** at **Grand Canyon University**. It demonstrates proficiency in **software engineering principles, DevOps integration, and applied machine learning deployment**.

**Author**: Imjot Bassi  
**Institution**: Grand Canyon University  
**Program**: Master of Science in Software Engineering  
**Project**: Capstone Milestone 5

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for full details.

---

## Contact

For questions, feedback, or collaboration opportunities:

* **GitHub**: [@imjbassi](https://github.com/imjbassi)
* **Repository**: [ai-model-packager](https://github.com/imjbassi/ai-model-packager)