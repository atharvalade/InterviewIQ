import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import pandas as pd

# Load your preprocessed dataset
df = pd.read_csv("DataSet.csv")  # Make sure this file contains your data

# Define your input columns (features)
input_columns = ["Communication Skills", "Confidence", "Knowledge about Destination Country", "Clarity of Purpose", "Financial Preparedness"]

# Define your target column (0 for visa not received, 1 for visa received)
target_column = "Visa Received"

# Split your data into input features (X) and target labels (y)
X = df[input_columns].astype(str)  # Convert all columns to strings
y = df[target_column].to_numpy()

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased")

# Tokenize your input data
X_encoded = tokenizer(list(X.to_records(index=False)), padding=True, truncation=True, return_tensors="tf", max_length=128, return_token_type_ids=False, return_attention_mask=True)

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((X_encoded.input_ids, X_encoded.attention_mask))

# Define your BERT model for classification
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
bert_output = model(input_ids, attention_mask=attention_mask)[0]
output = tf.keras.layers.Dense(1, activation="sigmoid")(bert_output[:, 0, :])

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model (you may need to split your data into training and validation sets)
model.fit(dataset, y, epochs=5, batch_size=32)

# Use the model to predict visa chances (you need to provide input data in the same format as training)
new_data = ["2", "4", "3", "3", "4"]
new_encoded = tokenizer(new_data, padding=True, truncation=True, return_tensors="tf", max_length=128, return_token_type_ids=False, return_attention_mask=True)
predictions = model.predict([new_encoded.input_ids, new_encoded.attention_mask])

# Display the prediction (0 means visa not received, 1 means visa received)
print(f"Visa Chance: {predictions[0][0]}")
