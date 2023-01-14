import tensorflow as tf
import numpy as np
import random

def play_game(model):
    board = np.zeros((3, 3))
    current_player = random.choice([1, 2])
    game_over = False
    winner = None

    while not game_over:
        if current_player == 1:
            # Use the model to predict the next move
            move_probs = model.predict(board.reshape(1, 9))
            next_move = np.argmax(move_probs)

            row, col = next_move // 3, next_move % 3

            # Make the move
            if board[row, col] != 0:
                row, col = random.choice(range(3)), random.choice(range(3))

                if board[row, col] != 0:
                    continue

                board[row, col] = current_player
            else:
                board[row, col] = current_player
        else:
            # Human player
            print_board(board)
            row = int(input("Enter row: ")) - 1
            col = int(input("Enter col: ")) - 1

            # Make the move
            if board[row, col] != 0:
                print("Invalid move")
                continue
            else:
                board[row, col] = current_player

        current_player = 3 - current_player
        game_over, winner = check_game_over(board)

    return board, winner, current_player

def print_board(board):
    for row in range(3):
        for col in range(3):
            if board[row, col] == 0:
                print(" ", end=" ")
            elif board[row, col] == 1:
                print("X", end=" ")
            elif board[row, col] == 2:
                print("O", end=" ")
            if col != 2:
                print("|", end=" ")
        print()
        if row != 2:
            print("---------")

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

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(9, input_shape=(9,), activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])

# Compile the model with a loss function and an optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.load_weights("weights_iteration.h5")

board, winner, current_player = play_game(model)

print_board(board)
