import numpy as np
import json
import pickle
import random
import nltk
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Initialize stemmer and lemmatizer
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# Load intents data
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?']

# Iterate through intents and tokenize patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Stem and lemmatize words, and remove ignore words
words = [lemmatizer.lemmatize(stemmer.stem(word.lower())) for word in words if word not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(set(classes))

# Create training data
training_data = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(stemmer.stem(word.lower())) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training_data.append(bag + output_row)

# Shuffle training data
random.shuffle(training_data)
training_data = np.array(training_data)

# Split training data into input and output
train_x = training_data[:, :len(words)]
train_y = training_data[:, len(words):]

# Define and compile the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')
print('Model saved.')
