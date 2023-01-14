import tensorflow as tf
import numpy as np
import random

def play_game(model1, model2):
    board = np.zeros((3, 3))
    current_player = 1
    game_over = False
    winner = None
    last_move = None

    while not game_over:
        if current_player == 1:
            model = model1
        else:
            model = model2

        # Use the model to predict the next move
        move_probs = model.predict(board.reshape(1, 9))
        next_move = np.argmax(move_probs)

        row, col = next_move // 3, next_move % 3

        while board[row][col] != 0:
            # If it is, choose a different action
            move_probs[0, next_move] = 0
            next_move = np.argmax(move_probs)

            row, col = next_move // 3, next_move % 3

        board[row, col] = current_player

        # Make the move
        # if board[row, col] != 0:
        #     row, col = random.choice(range(3)), random.choice(range(3))

        #     if board[row, col] != 0:
        #         continue

        #     board[row, col] = current_player
        # else:
        #     board[row, col] = current_player

        last_move = move_probs

        # Check if the game is over
        game_over, winner = check_game_over(board)

        # Switch to the other player
        current_player = 3 - current_player

    return board, last_move

def check_game_over(board):
    # Check for horizontal wins
    for row in range(3):
        if all(board[row, :] == 1):
            return True, 1
        elif all(board[row, :] == 2):
            return True, 2

    # Check for vertical wins
    for col in range(3):
        if all(board[:, col] == 1):
            return True, 1
        elif all(board[:, col] == 2):
            return True, 2

    # Check for diagonal wins
    if all(np.diag(board) == 1) or all(np.diag(np.fliplr(board)) == 1):
        return True, 1
    elif all(np.diag(board) == 2) or all(np.diag(np.fliplr(board)) == 2):
        return True, 2

    # Check for draw
    if np.count_nonzero(board) == 9:
        return True, 0

    # Otherwise, the game is not over
    return False, None

tf.config.experimental.set_visible_devices([], 'GPU')

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(9, input_shape=(9,), activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])

# Compile the model with a loss function and an optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.load_weights("weights_iteration.h5")

# Initialize the weights of the neural network with random values
# model.init_weights()

# Create a clone of the original model
clone_model = tf.keras.models.clone_model(model)

# Copy the weights of the original model to the clone

num_games = 1000

# Define the training loop
for i in range(num_games):
    print("Iteration: ", i)

    # Play a game of Tic-Tac-Toe against a clone of the AI
    game_result = play_game(model, clone_model)

    # Extract the board state and the move made from the game result
    board_state, move = game_result

    # Convert the board state and move to a format that can be used as input and output for the model
    x = np.array(board_state).reshape(1, 9)
    y = move

    # Update the weights of the neural network based on the outcome of the game
    model.train_on_batch(x, y)
    
    # Saving the model weights
    if i % 100 == 0 and i != 0:
        model.save_weights("weights_iteration.h5")
        print("Saved weights")

# After training, the neural network can be used to make moves in a game of Tic-Tac-Toe
