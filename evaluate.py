
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import joblib
from tensorflow.keras.models import load_model

# Załadowanie modelu i danych testowych
model = load_model("heart_failure_nn_model.h5")
X_test, y_test = joblib.load("test_data.pkl")

# Przewidywanie i ocena
y_pred_probs = model.predict(X_test).flatten()
y_pred = np.round(y_pred_probs)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_probs))

# WYkres
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc_score(y_test, y_pred_probs)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
