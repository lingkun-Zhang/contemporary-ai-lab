# 逻辑回归：查看训练、验证表现，再观察错例和类别加权的效果。
import numpy as np
import pandas as pd
from pathlib import Path

# 按脚本位置找数据，打开外层文件夹时也能运行。
DATA_DIR = Path(__file__).resolve().parent

def load_data():
    train_df = pd.read_csv(DATA_DIR / 'train_data.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

# 读取老师给的训练数据和无标签测试数据。
X_train, y_train, X_test_unlabeled = load_data()

# 先检查数量和文本格式。
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 查看一条文本及其标签。
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 无标签测试集用于最后预测，不参与调参。
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# 留出 20% 做验证；分层划分让两部分的类别比例接近。
from sklearn.model_selection import train_test_split

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

print("实际训练集：", len(X_fit))
print("验证集：", len(X_val))
print("老师提供的无标签测试集：", len(X_test_unlabeled))

# 检查分层划分后的类别分布。
from collections import Counter
print("训练集标签分布:",Counter(y_fit))
print("验证集标签分布:",Counter(y_val))

# TF-IDF 特征和逻辑回归。
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 词表和 IDF 只在训练部分拟合，验证部分沿用它们。
vectorizer = TfidfVectorizer(max_features=5000)
X_fit_tfidf = vectorizer.fit_transform(X_fit)
X_val_tfidf = vectorizer.transform(X_val)

# C 越大，正则化越弱；这里使用之前对照过的 C=100。
lr_model = LogisticRegression(
    penalty = 'l2',
    C = 100,
    max_iter=1000,
    random_state=42,
)

lr_model.fit(X_fit_tfidf, y_fit)
y_pred_lr = lr_model.predict(X_val_tfidf)

# 同时看准确率和交叉熵：前者看类别对错，后者看预测概率。
from sklearn.metrics import log_loss

# 准确率。
train_acc = accuracy_score(
    y_fit, lr_model.predict(X_fit_tfidf)
)
val_acc = accuracy_score(y_val, y_pred_lr)

# 这里的交叉熵不含 L2 惩罚；模型训练仍使用 L2。
train_loss = log_loss(
    y_fit,
    lr_model.predict_proba(X_fit_tfidf),
    labels=lr_model.classes_,
)
val_loss = log_loss(
    y_val,
    lr_model.predict_proba(X_val_tfidf),
    labels=lr_model.classes_,
)
print(f"本次 C={lr_model.C}")
print(f"逻辑回归训练集准确率：{train_acc:.2%}")
print(f"逻辑回归验证集准确率：{val_acc:.2%}")
print(f"逻辑回归训练集交叉熵损失：{train_loss:.4f}")
print(f"逻辑回归验证集交叉熵损失：{val_loss:.4f}")

# 找出验证集中分类错误的样本位置
wrong_indices = [
    i for i, (true, pred) in enumerate(zip(y_val, y_pred_lr))
    if true != pred
]

val_proba = lr_model.predict_proba(X_val_tfidf)

print(f"\n验证集分类错误：{len(wrong_indices)} / {len(y_val)} 条")

# 先查看前 5 条错误样本
for i in wrong_indices[:5]:
    print("\n" + "=" * 60)
    print(f"验证集样本位置：{i}")
    print(f"真实标签：{y_val[i]}，预测标签：{y_pred_lr[i]}")
    print(f"模型对预测类别的概率：{val_proba[i].max():.2%}")
    print("文本内容：")
    print(X_val[i][:1000])  # 先显示前 1000 个字符

from sklearn.metrics import f1_score

weight_results = []

print("\n权重 | 验证准确率 | Macro-F1 | 类别7召回率 | 验证损失")

for factor in [1.0, 1.5, 2.0]:
    # 类别 7 是这次加权的对象，不单独加权验证集中的错例。
    weights = np.where(y_fit == 7, factor, 1.0)

    # 平均权重保持为 1，避免总权重变化影响正则化的相对强度。
    weights /= weights.mean()

    weighted_model = LogisticRegression(
        C=100,
        max_iter=1000,
        random_state=42,
    )
    weighted_model.fit(
        X_fit_tfidf, y_fit, sample_weight=weights
    )

    pred = weighted_model.predict(X_val_tfidf)
    proba = weighted_model.predict_proba(X_val_tfidf)

    acc = accuracy_score(y_val, pred)
    macro_f1 = f1_score(y_val, pred, average="macro")
    recall_7 = (pred[y_val == 7] == 7).mean()
    loss = log_loss(
        y_val, proba, labels=weighted_model.classes_
    )

    weight_results.append(
        [factor, acc, macro_f1, recall_7, loss]
    )

    print(
        f"{factor:.1f} | {acc:.2%} | {macro_f1:.4f} | "
        f"{recall_7:.2%} | {loss:.4f}"
    )
