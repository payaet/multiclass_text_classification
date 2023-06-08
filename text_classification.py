"""
Exercise: Text Classification using LSTM
"""
#%%
#1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, datetime
from tensorflow import keras
from tensorflow.keras import callbacks
#%%
#2. Data loading
data = pd.read_csv("C:\\Users\\IT\\Desktop\\AI lesson\\DEEP LEARNING\\hands-on\\Assessment2\\ecommerceDataset.csv")
data.columns = ['category', 'text']
data.head()

# %%
#3. Data inspection
print(data.info())
print("-"*20)
print(data.describe())
print("-"*20)
print(data.isna().sum())
print("-"*20)
print(data.duplicated().sum())
#%%
data.dropna(inplace=True)
#data.drop_duplicates(inplace=True)
print(data.isna().sum())
#%%
#4. The text is the feature, the category is the label
feature = data['text'].values
label = data['category'].values
#%%
#5. Convert label into integers using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_processed = label_encoder.fit_transform(label)
# %%
#6. Data preprocessing
#(A) Remove unwanted strings from the data
#Import regular expression module
import re

def remove_unwanted_strings(review):
    for index, data in enumerate(review):
        # Anything within the <> will be removed 
        # ? to tell it dont be greedy so it won't capture everything from the 
        # first < to the last > in the document
        review[index] = re.sub('<.*?>', ' ', data) 
        review[index] = re.sub('[^a-zA-Z]',' ',data).lower().split()
    return review
feature_removed = remove_unwanted_strings(feature)
#%%
#7. Define some hyperparameters
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8
# %%
#8. Perform train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(feature_removed,label_processed,train_size=training_portion,random_state=12345)
# %%
#9. Perform tokenization
from tensorflow import keras

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,split=" ",oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
#%%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))
# %%
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
# %%
#10. Perform padding and truncating
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_tokens,maxlen=(max_length))
X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_tokens,maxlen=(max_length))

#%%
#11. Model development
#(A) Create the sequential model
model = keras.Sequential()
#(B) Create the input layer, in this case, it can be the embedding layer
model.add(keras.layers.Embedding(vocab_size,embedding_dim))
#(B) Create the bidirectional LSTM layer
model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim)))
#(C) Classification layers
model.add(keras.layers.Dense(embedding_dim,activation='relu'))
model.add(keras.layers.Dense(len(np.unique(y_train)),activation='softmax'))

model.summary()
# %%
#12. Model compilation
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# %%
# Create a TensorBoard callback object for the usage of TensorBoard
base_log_path = r"tensorboard_logs\ecommerce"
if not os.path.exists(base_log_path):
    os.makedirs(base_log_path)
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#13. Model training
history = model.fit(X_train_padded,y_train,validation_data=(X_test_padded,y_test),epochs=5,batch_size=64, callbacks=[tb])
# %%
#14. Model evaluation
print(history.history.keys())
# %%
#Plot accuracy graphs
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train accuracy","Test accuracy"])
plt.show()
#%%
from sklearn.metrics import f1_score
# Get the predicted probabilities for the test set
y_pred_probs = model.predict(X_test_padded)

# Convert probabilities to predicted labels
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='weighted')

print("F1 Score:", f1)


# %%
#15. Model deployment
test_string= ['Creative wall design has been placed as a decoration for the house.','Sony DVD PLAYER SN-3052V, latest technology with Sony DV-3052 HDMI 1080p Upscaling DVD Player with USB Playback']

test_string_removed = remove_unwanted_strings(test_string)
#%%
test_string_tokens = tokenizer.texts_to_sequences(test_string_removed)
#%%
test_string_padded = keras.preprocessing.sequence.pad_sequences(test_string_tokens,maxlen=(max_length))

# %%
y_pred = np.argmax(model.predict(test_string_padded),axis=1)
# %%
label_map = {0: "Books", 1: "Clothing&Accessories", 2: "Electronics", 3: "Household"}
predicted_categories = [label_map[i] for i in y_pred]
# %%
print(predicted_categories)

# %%
expected_categories = ['Household', 'Electronics']

for test, expected_category, predicted_category in zip(test_string, expected_categories, predicted_categories):
    print("Test String:", test)
    print("Expected Category:", expected_category)
    print("Predicted Category:", predicted_category)
    print("Labeling Correct:", expected_category == predicted_category)
    print()

# %%
#16. Save model and tokenizer
PATH = os.getcwd()
print(PATH)

# %%
from tensorflow.keras.models import load_model
model.save("C:\\Users\\IT\\Desktop\\AI lesson\\DEEP LEARNING\\hands-on\\Assessment2\\saved_models\\model")
# %%
#Check if the model can be loaded
model_loaded = keras.models.load_model("C:\\Users\\IT\\Desktop\\AI lesson\\DEEP LEARNING\\hands-on\\Assessment2\\saved_models\\model")
# %%
import json
# Save the tokenizer to a .json file
tokenizer_json = tokenizer.to_json()
tokenizer_path = os.path.join("saved_models", "tokenizer.json")

with open(tokenizer_path, "w") as json_file:
    json_file.write(tokenizer_json)

print("Tokenizer saved as tokenizer.json in the saved_models folder.")
# %%

