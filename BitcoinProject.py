# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. إنشاء هيكل المجلدات
folders = ['data/original', 'data/preprocessed', 'data/Results']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 2. تحميل البيانات الأصلية
btc_data = pd.read_csv('coin_Bitcoin.csv')
btc_data.to_csv('data/original/coin_Bitcoin.csv', index=False)

# 3. معالجة البيانات
btc_data['Target'] = (btc_data['Close'].shift(-1) > btc_data['Close']).astype(int)
print(btc_data.isnull().sum())
btc_data = btc_data.dropna()

# 4. تقسيم البيانات وحفظها
X = btc_data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = btc_data['Target']

print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42,
    shuffle=False
)

X_train.to_csv('data/preprocessed/X.csv', index=False)
X_test.to_csv('data/preprocessed/X_test.csv', index=False)
y_train.to_csv('data/preprocessed/Y_train.csv', index=False)
y_test.to_csv('data/preprocessed/Y_test.csv', index=False)

# 5. تحميل البيانات من الملفات المحفوظة
X_train_loaded = pd.read_csv('data/preprocessed/X.csv')
X_test_loaded = pd.read_csv('data/preprocessed/X_test.csv')
y_train_loaded = pd.read_csv('data/preprocessed/Y_train.csv')
y_test_loaded = pd.read_csv('data/preprocessed/Y_test.csv')

# 6. تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_loaded)
X_test_scaled = scaler.transform(X_test_loaded)

# 7. تعريف النماذج
models = {
    'ANN': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000),
    'SVM': SVC(kernel='rbf', probability=True),
    'NaiveBayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'DecisionTree': DecisionTreeClassifier(),
    'LinearReg': LinearRegression()
}

# 8. تدريب النماذج وحفظ النتائج
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train_loaded.values.ravel())
    
    if name == 'LinearReg':
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_scaled)
    
    # حفظ التنبؤات
    pred_df = pd.DataFrame({'Actual': y_test_loaded.values.ravel(), 'Predicted': y_pred})
    pred_df.to_csv(f'data/Results/predictions_{name}.csv', index=False)
    
    # حساب الدقة
    acc = accuracy_score(y_test_loaded, y_pred)
    results.append({'Model': name, 'Accuracy': acc})
    print(f'{name} Accuracy: {acc:.4f}')


# 9. إنشاء مقارنة الأداء
results_df = pd.DataFrame(results)

# أ) رسمة مقارنة الدقة
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Comparing the accuracy of models in classifying Bitcoin price trends', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.ylim(0, 1)
for i, acc in enumerate(results_df['Accuracy']):
    plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ب) رسمة violinplot للأخطاء
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

# ج) رسمة pairplot للبيانات الأصلية
sns.pairplot(btc_data[['Open', 'High', 'Low', 'Close', 'Volume']])
plt.suptitle('Analyzing the relationships between Bitcoin variables', y=1.02)
plt.savefig('data_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

print("تم الانتهاء بنجاح! جميع الملفات والرسومات محفوظة.")
