# **Eye Image Classification Using VGG19 and ResNet50**

## **Overview**
This repository contains the implementation of two deep learning models, **VGG19** and **ResNet50**, for classifying eye diseases using fundus images. These models aim to aid medical practitioners in automated diagnosis by leveraging state-of-the-art CNN architectures for accurate classification.

The project includes:
- Dataset preparation and preprocessing.
- Implementation of VGG19 and ResNet50 models.
- Model training and evaluation.
- Comparative analysis of the results.

---

## **Dataset**
The dataset consists of **5335 high-resolution images** divided into:
- **5318 color fundus images** (9 classes of eye diseases).
- **17 anterior segment images** (Pterygium).

### **Classes of Eye Diseases**
1. **Diabetic Retinopathy**  
2. **Glaucoma**  
3. **Macular Scar**  
4. **Optic Disc Edema**  
5. **Central Serous Chorioretinopathy (CSCR)**  
6. **Retinal Detachment**  
7. **Retinitis Pigmentosa**  
8. **Myopia**  
9. **Healthy**  
10. **Pterygium**  

### **Data Characteristics**
- **Resolution**: \(3900 \times 2600\) and \(2004 \times 1690\), resized to \(224 \times 224\) during preprocessing.  
- **Format**: JPEG.  

**Source**: The data was collected from Anawara Hamida Eye Hospital and B.N.S.B. Zahurul Haque Eye Hospital, Bangladesh, over 8 months (July 2023 - February 2024). 

For more details, refer to the **`[data_description.pdf](https://pdf.sciencedirectassets.com/311593/1-s2.0-S2352340924X00062/1-s2.0-S2352340924009417/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE0aCXVzLWVhc3QtMSJHMEUCICVQZGN%2FDTj%2FmpaJ0Q%2BlGI0mBgDl5ZW2FuKXYhmkC6UaAiEA5v3zCvt1xlyF90IeOB4jOZET%2B4XvG6styG4EW65NNGUqvAUI5f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDDcLE%2BgUSy4rPFZFkSqQBTYZtNR7bsA%2FnGsR2JHrK1F6Fo9E8kXtHTnBD%2BMOHWlBJHtL4ingKaG4RijkT%2FssFeCJAt9ZnEkAjVICkN5j5GtiF%2BQE5peDbJJo0TjVgCR3o8jtz6Q2OKrfFFkJonAL6tN%2BQl0uq6WCJAabVEe%2FieOoNwikPoeC5U178Zxlb1doer2i238h5zeJwe3ThKPyUoyXnsmEwmr7uW9sx4nzXnJauWTOqxUUy5pVcjvDJJkWVpH6WUS3ugmt0OODptxWm2h1DE0blHMPTqe5RFWw0GDLypz3iDHSt69Mzy744hu96nzzXhTomYA4nFQQ6fcUaXUWgvNl8Qt5wZfXX%2FJawytNxk98MEa7JApOZy3BMuN7IbZbiEV1xyiV%2BN1B441yHo5qJNE0v78qzNhgiKLt4YZYzZ3QyY55FoMWXyE9MespsSMPqG9VrAg3zAVviJwlCfSDnD8MbJKWctB3%2F%2FW6Xc5IG0D%2BsEU5%2BHZiVCPi8OLhDv4Vlk9c2QVcODQWODrI8t5MBkKBLYnrFsQS4t6UHlH7gdt%2BH%2FLEQL38QrX6M4ILAK%2FZR7Bk0tVx744LN7%2BFtYNcM3ZsONQIV0GxWgiMU7Je4mbmg9RPy7EMVWEu7Y%2BZpUYrble4ycr5QMmaRoeerUzgsjodeJK9d47UJ%2FeS3%2F8uENfC1SHjcgz6SmEDWwxbY%2B8L48twmELgCWX0nEODQZhN3yia1kurCWM1qPpKXTi9lZOZ2X6NQ3IZvVjWFWShheHw8cEi6k9tskevF55306ORzR88KvVwJ2yNK5kcptV7jo26Utt8DfUQOto25KnJYJp0lA7yNpjhWBmPogVqFZTP%2Bsqa1MVfLnOjA8cXvx3VyZJzdA7RFXB73BrHwe9bMM%2FQiroGOrEBGaU7AjzJ6pBUUGT3SyaSVcmquuJJN8aLYzDVX87yrXkUbI7xImQSJ8WI8tPTLaLrAnEYVd%2FvX0YvE0uWb0OmuuYYMi7DmKMG0769vaEW6VoPQPyxTNPNm70drVzhxU7OnD1OV5zuU%2F5y99CMEq0X6dBodB0VwMXj7Tu5fJ5%2BY3DNcm70qG%2FvGTV9NuF95PHrQHnmbBYMbcOHDynBsDjIcoFDbsW8rtNPyuo5a0VTiL0y&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241124T044220Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYVQPBTTIL%2F20241124%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=fd6add72c36248a5f5824cd4c6d3b0721a4657852aae0f1fe940fcb1c0174aa3&hash=fdc15ae763c3de6a76eacef9a60a79e212583f746466145ee4d75126ceb85d16&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2352340924009417&tid=spdf-eafb357b-aeca-41e8-a973-0d7de37e4356&sid=9682d3f5337216445338a0c7aa575e8ff11egxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f0e5e0151540d075650&rr=8e76bb531caf5505&cc=in)`** file.

---

## **Deep Learning Models**

### **1. VGG19**
- **Architecture**: A 19-layer CNN with uniform \(3 \times 3\) convolutional layers, ReLU activations, and max pooling.
- **Key Features**:
  - Simplicity and uniform design.
  - Pre-trained weights (ImageNet) for faster convergence.  

### **2. ResNet50**
- **Architecture**: A 50-layer CNN utilizing residual connections to prevent vanishing gradient problems.
- **Key Features**:
  - Skip connections to preserve gradient flow.
  - Efficient bottleneck layers (\(1 \times 1\), \(3 \times 3\), \(1 \times 1\)).  

---

## **Training and Evaluation**
### **Data Preprocessing**
- **Augmentation**: Rotation, flipping, and normalization to enhance generalization.
- **Input Size**: \(224 \times 224 \times 3\).

### **Training Setup**
| Model    | Epochs | Batch Size | Optimizer | Loss Function           |
|----------|--------|------------|-----------|-------------------------|
| **VGG19**   | 50     | 32         | Adam      | Categorical Cross-Entropy |
| **ResNet50**| 10     | 16         | Adam      | Categorical Cross-Entropy |

### **Hardware Used**
- **Processor**: AMD Ryzen 7 6800H.

### **Evaluation Metrics**
1. **Accuracy**  
2. **Precision**  
3. **Recall**  
4. **F1-Score**  
5. **Confusion Matrix**

---

## **Results**

| Metric         | **VGG19** | **ResNet50** |
|----------------|-----------|--------------|
| **Accuracy**   | 55%       | 85%          |
| **Precision**  | 56%       | 86%          |
| **Recall**     | 55%       | 85%          |
| **F1-Score**   | 55%       | 85%          |
| **Training Time**| 28 Hours | 12 Hours     |

### **Key Observations**
- **ResNet50** significantly outperforms **VGG19** across all metrics due to its residual connections and efficient architecture.  
- **VGG19** shows limitations in deeper feature extraction and slower training.  
- **ResNet50** achieves higher accuracy and faster convergence with a shorter training time.

---

=

---

### **1. Prepare the Dataset**
- Download the dataset from the **Mendeley Data Repository**.
- Place the dataset in the `data/` directory.

### **2. Train the Models**
- To train VGG19:
  ```bash
  python models/vgg19.py
  ```
- To train ResNet50:
  ```bash
  python models/resnet50.py
  ```

### **3. Evaluate the Models**
Use the notebooks in the `notebooks/` directory to generate evaluation metrics and visualize results.

---

## **Recommendations**
1. **Use ResNet50** for production due to its superior accuracy and shorter training time.  
2. **Integrate Grad-CAM** visualizations to enhance interpretability of predictions.  
3. Explore **ensemble learning** for further performance improvements.

---

## **References**
1. **Dataset**: [Mendeley Data Repository](https://data.mendeley.com/datasets/s9bfhswzjb/1).  
2. **Model Architectures**: Derived from official implementations of VGG19 and ResNet50.

---
