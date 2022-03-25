English · [简体中文](./README_zh.md) 

---

## 2021 Sohu Campus Text Matching Algorithm Competition Program

> [Official Website](https://www.biendata.xyz/competition/sohu_2021/)

We are the team **BERloomers**, ranked *9th* in the preliminary round and *15th* in the rematch.

Members: KennyWu (kennywu96@163.com), null_li (lizx3845@163.com).

This open source solution was redesigned after the competition, and the offline validation set F1 is about 0.7887.

We welcome your comments and corrections.

![](figure/sohu2021.png)

## 1. Scheme

### 1.1 Task
The contestants need to correctly determine whether two texts match or not.

<a href="https://www.biendata.xyz/competition/sohu_2021/data/">The data</a> is divided into two files, A and B, which have different matching criteria.
The A and B files are divided into **short text matching**, **short and long text matching** and **long and long text matching**.

The A file has a broader matching criterion, where two paragraphs are considered to be a match if they are on the same topic, 
while the B file has a stricter matching criterion, where two paragraphs are considered to be a match only if they are on the same event.

### 1.2 Data example
```python
# A short-short sample
{
    "source": "小艺的故事让爱回家2021年2月16日大年初五19：30带上你最亲爱的人与团团君相约《小艺的故事》直播间！",
    "target": "  香港代购了不起啊，宋点卷竟然在直播间“炫富”起来",
    "labelA": "0"
}

# B short-short sample
{
    "source": "让很多网友好奇的是，张柏芝在一小时后也在社交平台发文：“给大家拜年啦。”还有网友猜测：谢霆锋的经纪人发文，张柏芝也发文，并且配图，似乎都在证实，谢霆锋依旧和王菲在一起，而张柏芝也有了新的恋人，并且生了孩子，两人也找到了各自的归宿，有了自己的幸福生活，让传言不攻自破。",
    "target": "  陈晓东谈旧爱张柏芝，一个口误暴露她的秘密，难怪谢霆锋会离开她", 
    "labelB": "0"
}
```

### 1.3 Scheme design
In order to learn as much information as possible from the data while also taking into account standards A, B, 
and the three minor classification criteria, 
our plan is based on a multi-task learning framework, 
sharing some parameters for representation learning, 
and then designing task-specific classifiers for label prediction.

#### 1.3.1 Model
The framework is designed based on BERT for an interactive model, using BERT to obtain vector representations of source-target pairs. 
The overall structure of this plan is shown in the figure below:
![overall architecture](figure/model.png)

#### 1.3.2 Encoding
In this plan, the results from the last 3 layers are used for learning downstream tasks.
Moreover, considering the characteristic of this competition being divided into 6 sub-tasks, we introduce the concept of Special Tokens.
- Six Type Tokens are proposed to guide the representation learning of text:

    | Token | Task type     |
    | --- |---------------|
    | SSA | short-short A |
    | SSB | short-short B |
    | SLA | short-long A  |
    | SLB | short-long B  |
    | LLA | long-long A   |
    | LLB | long-long B   |

- Use `[<S>]` and `[</S>]` to distinguish the source, and `[<T>]` and `[</T>]` to distinguish the target. (The corresponding special tokens have been added to the vocab.txt file under models/*.)

#### 1.3.3 Multi-task learning
In order to better learn the representation of `Type Token` and to assist in the learning of `Task-Attentive Classifier`, 
we have proposed a data type label prediction task, 
which is to determine the type of task the current input belongs to based on the representation of `Type Token`.

#### 1.3.4 Task-Attentive Classifier
Adhering to the philosophy of "harmony in diversity," 
tasks A and B each independently employ a `Task-Attentive Classifier`. 
Meanwhile, the representation of Type Token is passed as additional conditional information into the `Classifier` for attention computation, 
thereby obtaining type-specific features for label prediction.

![task attentive classifier](figure/task_attentive.png)

### 1.4 Competitive tricks

#### 1.4.1 Data segmentation
The training data used in this plan includes both the training set provided for the semifinals and all the data provided in the preliminary round. 
The validation set provided for the semifinals is used to evaluate the performance of the model.

#### 1.4.2 Model fusion
Based on the F1 scores of three models on the offline validation set, 
different weights were assigned, 
and the optimal weight combination was found through **automatic search** to achieve the best performance offline. 
In this solution, the `WoBERT` model used comes from the customized PyTorch-based pre-trained model loading framework created by member KennyWu, 
<a href="https://github.com/KKenny0/torchKbert">torchKbert</a>, 
and we express our gratitude to Zhuiyi Technology for their open-sourced <a href="https://github.com/ZhuiyiTechnology/WoBERT">WoBERT</a>.

| 模型               |                     链接                     |
|:-----------------|:------------------------------------------:|
| BERT-wwm-ext     | https://github.com/ymcui/Chinese-BERT-wwm  |
| RoBERTa-wwm-ext  | https://github.com/ymcui/Chinese-BERT-wwm  |
| WoBERT           | https://github.com/ZhuiyiTechnology/WoBERT |


#### 1.4.3 Adversarial training、EMA和 Focal Loss

### 1.5 Improvements compared to the proposal submitted in the competition
The open-source solution for this round has been improved over the previous submission in three aspects: **data segmentation**, **model architecture**, and **model ensemble**.
- **Data segmentation**：Expand the training set: the training set provided for the second round --> the training set provided for the second round + all the data from the preliminary round.
- **Model architecture**：The network structure has been redesigned to improve the method of Task-specific encoding.
- **Model fusion**：The plan submitted for the second round of the competition used the fusion of `BERT-wwm-ext` and `ERNIE-1.0` models. This plan, however, employs the fusion of `BERT-wwm-ext`, `RoBERTa-wwm-ext`, and `WoBERT`.

## 2. 复赛反思
The preparation for the semi-final was somewhat hasty, 
and the plan we submitted for the semi-final had many shortcomings. 
Therefore, after the competition, we reviewed the entire process and improved the plan. 
Although we did not make it into the Top 10 in the semi-final, 
this competition was still a valuable experience for us.
