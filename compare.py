import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def build_comparison_table(ml_results,ann_result):
    rows=[]

    for name, res in ml_results.items():
        rows.append({
            'Model': name,
            'Accuracy (%)': round(res['accuracy']  * 100,2),
            'Training Time (s)': round(res['train_time'],2),
            'Learns Features':"No",
            'Type' : "Traditional ML"

        })

    rows.append({
            'Model' : 'ANN',
            'Accuracy (%)': round(ann_result['accuracy']*100,2),
            'Training Time (s)': round(ann_result['train_time'],2),
            'Learns Features':"Yes",
            'Type' : "Deep Learning"
        })
    return rows


def plot_accuracy_comparison(comparison_table, save_path):
    models = [r['Model'] for r in comparison_table]
    accs   = [r['Accuracy (%)'] for r in comparison_table]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accs, color=colors[:len(models)])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison — Fashion MNIST')
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc}%', ha='center', fontweight='bold')
    plt.ylim(75, 95)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_ann_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('ANN Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(history['accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('ANN Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_sample_predictions(X_test, y_test, all_results, class_names, save_path):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    indices = np.random.choice(len(X_test), 10, replace=False)
    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        ax.imshow(X_test[idx], cmap='gray')
        actual = class_names[y_test[idx]]
        preds = {name: class_names[res['y_pred'][idx]] for name, res in all_results.items()}
        title = f"Actual: {actual}\n"
        title += "\n".join([f"{n}: {p}" for n, p in preds.items()])
        ax.set_title(title, fontsize=7)
        ax.axis('off')
    plt.suptitle('Sample Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
        