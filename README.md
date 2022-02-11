[![Generic badge](https://img.shields.io/badge/Made_With-Python-<COLOR>.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/ALGO-RF-red.svg)](https://shields.io/)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
![visitor badge](https://visitor-badge.glitch.me/badge?page_id=o-ikne.task0-data-project)

# task0-data-project

## **Overview**
The purpose of this project is to propose models for the shared-task task 0 problem. To do so, we will use the development data to design two models (a neural and a non neural) that given a lemma and a set of morphological attributes return the corresponding form.

## **Data**
The data used in this project are available [here](https://github.com/sigmorphon2020/task0-data). These are developmental languages and surprise languages for the SIGMORPHON 2020 task 0 (Typologically Diverse Morphlogical Inflection). The specificity of this dataset lies in the fact that it contains several language families, each of which contains a different number of languages. Here is an example from swahli (swa) language.

| lemma | form | morphological attributes |
|-------|------|--------------------------|
| piga |	amepiga	| V;PRF;FIN;IND;SG;3;PST |
| kamilisha	| wangekamilisha |	V;FIN;COND;PL;3;PRS |
| fuata	| walifuata |	V;FIN;IND;PL;3;PST |
| tengeneza |	nitatengeneza |	V;FIN;IND;SG;1;FUT |
| uwa	| anauwa | V;DEF;FIN;IND;SG;3;PRS |

## **Evaluation**

### **Bag-Of-Words Approach (BOW)**
```
python bag_of_words.py train_file test_file verbose=1
```
### **Prefix-Suffix Approach (PS)**
```
python prefix_suffix.py train_file test_file verbose=1
python prefix_suffix_improved.py train_file test_file verbose=1
```

### **Root-Prefix-Suffix (RPS)**
```
python root_prefix_suffix.py train_file test_file verbose=1
```


## **Installation**
To try our implementation in your own machine, you need to download the data from [task0-data](https://github.com/sigmorphon2020/task0-data) and install the following requirements:
```
pip install -r requirements.txt
```

## **References**

>@article{vylomova2020sigmorphon, title={SIGMORPHON 2020 Shared Task 0: Typologically Diverse Morphological Inflection}, author={Vylomova, Ekaterina and White, Jennifer and Salesky, Elizabeth and Mielke, Sabrina J and Wu, Shijie and Ponti, Edoardo and Maudslay, Rowan Hall and Zmigrod, Ran and Valvoda, Josef and Toldova, Svetlana and others}, journal={SIGMORPHON 2020}, pages={1}, year={2020}}

