import numpy as np

class RNN(object):

    def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):

        self.hidden_size = hidden_size # h
        self.vocab_size = vocab_size # x(t)
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        self.W1 = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (hidden_size, vocab_size))
        self.W2 = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (hidden_size, hidden_size))
        self.W3 = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (vocab_size, hidden_size))
        self.b1 = np.zeros((hidden_size, 1)) # h
        self.b2 = np.zeros((vocab_size, 1)) # o
    
    
    def forward(self, inputs, memory):

        x, h, o, y = {}, {}, {}, {}
        h[-1] = np.copy(memory)

        for t in range(len(inputs)):
            char_index = inputs[t]  # Make sure inputs[t] is an integer representing the character index
            x[t] = np.zeros((self.vocab_size, 1))
            x[t][char_index] = 1 
            h[t] = np.tanh(np.dot(self.W1, x[t]) + np.dot(self.W2, h[t-1]) + self.b1)
            o[t] = np.dot(self.W3, h[t]) + self.b2
            y[t] = self.softmax(o[t])

        return x, h, y
    

    def softmax(x):

        e = np.exp(x - np.max(x))
        
        return e/np.sum(e)
    

    def loss(self, p, targets):

        return sum(-np.log(p[t][targets[t][0]]) for t in range(len(self.seq_length)))
    

    def backprop(self, x, h, y, targets):

        dW1, dW2, dW3 = np.zeros_like(self.W1), np.zeros_like(self.W2), np.zeros_like(self.W3)
        db1, db2 = np.zeros_like(self.b1), np.zeros_like(self.b2)
        dhnext = np.zeros_like(h[0])

        for t in reversed(range(self.seq_length)):

            dy = np.copy(y[t])
            dy[targets[t]] -= 1

            dw3 += np.dot(dy, h[t].T)
            db2 += dy

            dh = np.dot(self.W3.T, dy) + dhnext

            dhrec = (1 - h[t]*h[t])*dh

            db1 += dhrec
            dW1 += np.dot(dhrec, x[t].T)
            dW2 += np.dot(dhrec, h[t-1].T)

            dhnext = np.dot(self.W2.T, dhrec)

        for dparam in [dW1, dW2, dW3, db1, db2]:
            
            np.clip(dparam, -5, 5, out=dparam)
        
        return dW1, dW2, dW3, db1, db2
    

    def update_weights(self, dW1, dW2, dW3, db1, db2):

        for param, dparam in zip([self.W1, self.W2, self.W3, self.b1, self.b2], [dW1, dW2, dW3, db1, db2]):

            param += -self.learning_rate*dparam
    

input_sequences = ['hello worl', 'python is ', 'amazing!!!']
target_sequences = ['ello world', 'ython is a', 'mazing!!! ']

# Hyperparameters
hidden_size = 64
vocab_size = 256  # Assuming characters are represented by their ASCII values (0 to 255)
seq_length = 10
learning_rate = 0.01
num_epochs = 100
batch_size = 1

# Create and train the RNN model
model = RNN(hidden_size, vocab_size, seq_length, learning_rate)

def one_hot_encoding(char_indices, vocab_size):
    one_hot_vector = np.zeros((vocab_size, 1))
    one_hot_vector[char_indices] = 1
    return one_hot_vector

def train_rnn(model, input_sequences, target_sequences, num_epochs, batch_size):
    num_batches = len(input_sequences) // batch_size

    for epoch in range(num_epochs):
        total_loss = 0

        # Shuffle the data for each epoch
        indices = np.arange(len(input_sequences))
        np.random.shuffle(indices)

        for batch_num in range(num_batches):
            # Get the batch data
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_inputs = [input_sequences[i] for i in batch_indices]
            batch_targets = [target_sequences[i] for i in batch_indices]

            # Initialize memory for the first batch element
            memory = np.zeros((model.hidden_size, 1))

            for t in range(seq_length):
                x, h, y = model.forward([one_hot_encoding(ord(seq[t]), vocab_size) for seq in batch_inputs], memory)
                loss = model.loss(y, [ord(target[t]) for target in batch_targets])
                total_loss += loss

                dW1, dW2, dW3, db1, db2 = model.backprop(x, h, y, [ord(target[t]) for target in batch_targets])
                model.update_weights(dW1, dW2, dW3, db1, db2)

                # Update memory for the next time step
                memory = h[model.seq_length - 1]

        # Print average loss for the epoch
        average_loss = total_loss / (num_batches * model.seq_length)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")


# Training the RNN model
train_rnn(model, input_sequences, target_sequences, num_epochs, batch_size)

# Generating a response
def generate_response(model, input_sequence, num_chars=50):
    response = input_sequence
    memory = np.zeros((model.hidden_size, 1))

    for _ in range(num_chars):
        x, h, y = model.forward([one_hot_encoding(ord(char), vocab_size) for char in response[-seq_length:]], memory)
        next_char_idx = np.random.choice(range(vocab_size), p=y[model.seq_length - 1].ravel())
        response += chr(next_char_idx)

        # Update memory for the next iteration
        memory = h[model.seq_length - 1]

    return response

# Example usage:
input_sequence = 'hello'
generated_response = generate_response(model, input_sequence, num_chars=50)
print("Generated Response:", generated_response)