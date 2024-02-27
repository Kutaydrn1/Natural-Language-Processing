import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load and preprocess the dataset
# (Steps for loading the dataset are not mentioned here.)
max_words = 10000
maxlen = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)

# Create the model
model = Sequential([
    Embedding(max_words, 32, input_length=maxlen),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)
