import os
import numpy as np
import json
from sklearn.metrics import confusion_matrix
from data_loader import load_data,explore_data,prepare_data, CLASS_NAMES
from ml_models import train_logistic_regression,train_svm,train_random_forest
from ann_model import build_ann,train_ann
from compare import (build_comparison_table,plot_accuracy_comparison,plot_ann_training_curves,plot_confusion_matrix,plot_sample_predictions)

os.makedirs('results/plots',exist_ok=True)

#Load data
print("STEP 1: Loading Fashion MNIST")
(X_train,y_train),(X_test,y_test) = load_data()
explore_data(X_train,y_train)

#Preapre data
X_train_norm,X_test_norm,X_train_flat,X_test_flat = prepare_data(X_train,X_test)

#Train ML models
print("\n STEP 2: Training ML models")

print("\nTraining Logistic Regression...")
lr = train_logistic_regression(X_train_flat,y_train,X_test_flat,y_test)
print(f"  Accuracy: {lr['accuracy']*100:.2f}% | Time: {lr['train_time']:.1f}s")

print("\nTraining SVM...")
svm = train_svm(X_train_flat,y_train,X_test_flat,y_test)
print(f"  Accuracy: {svm['accuracy']*100:.2f}% | Time: {svm['train_time']:.1f}s")

print("\nTraining Random Forest...")
rf = train_random_forest(X_train_flat,y_train,X_test_flat,y_test)
print(f"  Accuracy: {rf['accuracy']*100:.2f}% | Time: {rf['train_time']:.1f}s")

ml_results={
    'Logistic Regression':lr,
    'SVM':svm,
    'Random Forest':rf
}

#Train ANN
print("\n STEP 3: Training ANN")
ann_model = build_ann()
ann_model.summary()
ann = train_ann(ann_model,X_train_norm,y_train,X_test_norm,y_test,epochs=20)
print(f"\n  ANN Accuracy: {ann['accuracy']*100:.2f}% | Time : {ann['train_time']:.1f}s")

# Compare & generate plots
print("\n STEP 4: Generating Comparisions and Plot")
table = build_comparison_table(ml_results, ann)

plot_accuracy_comparison(table, 'results/plots/accuracy_comparison.png')
print("  ✅ Accuracy comparison chart saved")
plot_ann_training_curves(ann['history'], 'results/plots/ann_training_curves.png')
print("  ✅ ANN training curves saved")

all_results ={ **ml_results,'ANN':ann}

for name,res in all_results.items():
    cm= confusion_matrix(y_test,res['y_pred'])
    fname=name.lower().replace(" ","_")
    plot_confusion_matrix(cm,CLASS_NAMES, f'{name} Confusion Matrix',
    f'results/plots/{fname}_cm.png')
    print("  ✅ Confusion matrices saved")

    plot_sample_predictions(X_test,y_test,all_results,CLASS_NAMES,
    'results/plots/sample_predictions.png')
    print("  ✅ Sample predictions saved")


# 6. Save metrics to JSON
save_data = {}
for name, r in all_results.items():
    save_data[name] = {
        'accuracy': r['accuracy'],
        'train_time': r['train_time'],
        'report': r['report'],
        'confusion_matrix': r['confusion_matrix']
    }
with open('results/ml_results.json', 'w') as f:
    json.dump(save_data, f, indent=2)
with open('results/ann_history.json', 'w') as f:
    json.dump(ann['history'], f)
# Save predictions for Streamlit app
np.save('results/y_pred_all.npy', {name: res['y_pred'] for name, res in all_results.items()})
print("\n" + "=" * 50)
print("✅ ALL DONE! Results saved to results/")
print("=" * 50)
print("\nNow run:  streamlit run app.py")
