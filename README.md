# WAlign: Weight-Guided Alignment for Open-Set Anomaly Detection


## Dataset Settings

The specific settings for the five datasets used in our experiments are as follows:

### **FMNIST**
The FMNIST dataset contains grayscale images from ten fashion categories.  
- **Seen normal:** *T-shirt*, *Dress*, *Coat*  
- **Unseen normal:** *Shirt*  
- **Seen anomaly:** *Sneaker*, *Sandal*  
- **Unseen anomaly:** *Ankle Boot*, *Trouser*

---

### **MNIST-C**
MNIST-C is a handwritten digit dataset with various perturbations.  
- **Seen normal:** Digit 0 under *no perturbation*, *brightness perturbation*, and *shot-noise perturbation*  
- **Unseen normal:** Digit 0 under *fog perturbation*  
- **Seen anomaly:** Digit 1 under *no perturbation* and *brightness perturbation*  
- **Unseen anomaly:** Digit 1 under *shot-noise perturbation* and digit 7 under *no perturbation*

---

### **Yelp**
Yelp is a textual dataset comprising business reviews with ratings ranging from 1-star to 5-star.  
- **Normal:** 4-star and 5-star reviews  
- **Seen anomaly:** 1-star reviews  
- **Unseen anomaly:** 2-star reviews

---

### **NSL-KDD**
NSL-KDD is a tabular dataset used for network intrusion detection.  
- **Seen anomaly:** *DoS*, *R2L*  
- **Unseen anomaly:** Remaining anomaly classes

---

### **NB15**
NB15 is another network intrusion detection dataset that includes seven anomalous categories.  
- **Seen anomaly:** *Backdoor*, *DoS*, *Generic*  
- **Unseen anomaly:** Remaining anomaly classes

---

## 🔗 Original Dataset Links

| Dataset | Link |
|:--------:|:-----|
| **FMNIST** | [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) |
| **MNIST-C** | [https://github.com/google-research/mnist-c](https://github.com/google-research/mnist-c) |
| **Yelp** | [https://huggingface.co/datasets/Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full) |
| **NB15** | [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| **NSL-KDD** | [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html) |
