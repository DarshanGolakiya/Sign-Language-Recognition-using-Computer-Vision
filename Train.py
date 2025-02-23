import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, confusion_matrix
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

data_padded = pad_sequences(data, padding='post', dtype='float32')

x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
f1 = f1_score(y_test, y_predict, average='weighted')

accuracy = accuracy_score(y_test, y_predict)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'F1 Score: {f1:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

cm = confusion_matrix(y_test, y_predict)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predict, alpha=0.6, color='blue')
plt.title('Scatter Plot: True Labels vs Predicted Labels')
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.grid(True)

plt.show()

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
