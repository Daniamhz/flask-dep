import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# إنشاء بيانات عشوائية
X, y = make_classification(n_samples=100, n_features=9, n_classes=2, random_state=42)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# إنشاء مقياس ونموذج
scaler = StandardScaler()
model = SVC()

# إنشاء نموذج أنابيب (Pipeline)
pipeline = make_pipeline(scaler, model)

# تدريب النموذج
pipeline.fit(X_train, y_train)

# حفظ النموذج والمقياس
joblib.dump(pipeline, 'model_and_scaler.pkl')

# التحقق من النموذج
print(f'Model accuracy: {pipeline.score(X_test, y_test)}')

import joblib
import numpy as np

# تحميل النموذج والمقياس
pipeline = joblib.load('model_and_scaler.pkl')

# إدخال بيانات اختبار
data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

# إجراء التنبؤ
prediction = pipeline.predict(data)

print(f'Prediction: {prediction[0]}')
