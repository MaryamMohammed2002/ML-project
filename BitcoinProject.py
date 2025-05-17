import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. إنشاء المجلدات
folders = ['data/original', 'data/preprocessed', 'data/Results']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 2. تحميل البيانات 
btc_data = pd.read_csv('coin_Bitcoin.csv')
btc_data.to_csv('data/original/coin_Bitcoin.csv', index=False)

# 3. عرض القيم الناقصة والوصف الإحصائي
# ==========================================================
print("Missing Values:")
print(btc_data.isnull().sum())
print("\nData Description:")
print(btc_data.describe())


# 4. إنشاء العمود الهدف ومعالجة البيانات
btc_data['Target'] = (btc_data['Close'].shift(-1) > btc_data['Close']).astype(int)
btc_data.dropna(inplace=True)


# 5. اختيار الخصائص وتقسيم البيانات
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = btc_data[features]
y = btc_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

X_train.to_csv('data/preprocessed/X.csv', index=False)
X_test.to_csv('data/preprocessed/X_test.csv', index=False)
y_train.to_csv('data/preprocessed/Y_train.csv', index=False)
y_test.to_csv('data/preprocessed/Y_test.csv', index=False)

# 6. تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. تعريف النماذج (بدون Linear Regression)
models = {
    'ANN': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000),
    'SVM': SVC(kernel='rbf', probability=True),
    'NaiveBayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'DecisionTree': DecisionTreeClassifier()
}

results = []
confusion_data = []

# 8. تدريب النماذج + حفظ التوقعات + إعداد بيانات مصفوفة الالتباس
for name, model in models.items():
    model.fit(X_train_scaled, y_train.values.ravel())
    y_pred = model.predict(X_test_scaled)

    pd.DataFrame({'Actual': y_test.values.ravel(), 'Predicted': y_pred}).to_csv(
        f'data/Results/predictions_{name}.csv', index=False
    )

    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})
    print(f'{name} Accuracy: {acc:.4f}')
    
    cm = confusion_matrix(y_test, y_pred)
    confusion_data.append((name, cm))

# 9. رسم جميع مصفوفات الالتباس في شبكة واحدة
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Confusion Matrices for All Models', fontsize=16)

for idx, (name, cm) in enumerate(confusion_data):
    row, col = divmod(idx, 3)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[row, col], cmap='Blues', colorbar=False)
    axes[row, col].set_title(name)

if len(confusion_data) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('confusion_all_models.png')
plt.show()
# ==========================================================
# 10. رسم مقارنة دقة النماذج - barplot
# ==========================================================
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))

# تعليق توضيحي :
# هذا الرسم يستخدم للمقارنة بين دقة النماذج بشكل مباشر. يعطي نظرة سريعة على أفضل نموذج أداءً.
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='Blues_d')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(results_df['Accuracy']):
    plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
barplot_path = 'barplot_accuracy.png'
plt.savefig(barplot_path)
plt.show()

# ==========================================================
# 11. رسم pairplot - تحليل العلاقات الثنائية
# ==========================================================

# هذا النوع من الرسوم يوضح العلاقة بين كل زوج من المتغيرات. مفيد لاكتشاف الأنماط والترابط.
pairplot_path = 'pairplot.png'
sns.pairplot(btc_data[features])
plt.savefig(pairplot_path)
plt.show()

# ==========================================================
 # 12. رسم violinplot - توزيع الأخطاء لكل نموذج
# ==========================================================

errors = []
for name in models.keys():
    pred_df = pd.read_csv(f'data/Results/predictions_{name}.csv')
    error = pred_df['Actual'] - pred_df['Predicted']
    errors.append(pd.
    DataFrame({'Model': name, 'Error': error}))
    
error_df = pd.concat(errors)
plt.figure(figsize=(12, 6))
sns.violinplot(x='Model', y='Error', data=error_df, inner='quartile', palette='coolwarm')
plt.title('Distribution of errors across models', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.savefig('models_errors.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================================
# 13. رسم الهيستوغرامات - توزيع فردي لكل خاصية
# ==========================================================

# الهيستوغرام يساعد على فهم شكل توزيع كل متغير بشكل منفصل.
histogram_path = 'histograms.png'
plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(btc_data[col], bins=30, kde=True, color='steelblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig(histogram_path)
plt.show()

# ==========================================================
# 14. رسم خريطة الحرارة - تحليل الترابط بين الخصائص
# ==========================================================

# هذا الرسم يوضح مدى ترابط كل خاصيتين رقميتين باستخدام اللون والقيمة.
heatmap_path = 'heatmap.png'
plt.figure(figsize=(8, 6))
sns.heatmap(btc_data[features].corr(), annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
