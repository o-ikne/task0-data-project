# task0-data-project

### **Overview**
The purpose of this project is to propose models for the shared-task task 0 problem. To do so, we will use the development data to design two models (a neural and a non neural) that given a lemma and a set of morphological attributes return the corresponding form.

### **Data**
The data used in this project are available [here](https://github.com/sigmorphon2020/task0-data). These are developmental languages and surprise languages for the SIGMORPHON 2020 task 0 (Typologically Diverse Morphlogical Inflection). The specificity of this dataset lies in the fact that it contains several language families, each of which contains a different number of languages.

### **Evaluation**

#### **Bag-of-words**
```
python bag_of_words_pipeline.py train_file test_file verbose
```
#### **OneHot encoding**


### **Installation**
To try our implementation in your own machine you need to install the following requirements:
```
pip install -r requirements.txt
```

### **References**
```
@article{vylomova2020sigmorphon, title={SIGMORPHON 2020 Shared Task 0: Typologically Diverse Morphological Inflection}, author={Vylomova, Ekaterina and White, Jennifer and Salesky, Elizabeth and Mielke, Sabrina J and Wu, Shijie and Ponti, Edoardo and Maudslay, Rowan Hall and Zmigrod, Ran and Valvoda, Josef and Toldova, Svetlana and others}, journal={SIGMORPHON 2020}, pages={1}, year={2020}}
```
