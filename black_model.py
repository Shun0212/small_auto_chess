import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_attack_data(file_path='minmax_chess_ver6.json.json'):
    game_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            game_id = data['game_id']
            if game_id not in game_data:
                game_data[game_id] = {'scores': [], 'turn': 0}
            if 'move_score' in data and 'board_score' in data and data['piece'].startswith('b'):
                move_score = data['move_score']
                board_score = data['board_score']
                white_king_threat = data['White_king_threat']
                threatened_pieces = sum(data['threatened_pieces'].values()) if 'threatened_pieces' in data else 0
                can_move_score = data['can_move_score']['black_can_move_score']
                can_get_score = data['can_move_score']["black_can_get_score"]
                scores = [move_score, board_score, white_king_threat, threatened_pieces, can_move_score, can_get_score]
                game_data[game_id]['scores'].append(scores)

    X_scores, y = [], []
    for game_id, game in game_data.items():
        if len(game['scores']) > 1:
            for i in range(0, len(game['scores']) - 1):
                current_score = game['scores'][i]
                next_white_king_threat = game['scores'][i + 1][2]
                X_scores.append(current_score)
                y.append(next_white_king_threat)

    return np.array(X_scores), np.array(y)

X_scores, y = load_attack_data()

if X_scores.size == 0 or y.size == 0:
    raise ValueError("No data available for training.")

X_scores_train, X_scores_test, y_train, y_test = train_test_split(X_scores, y, test_size=0.2, random_state=42)

input_scores = Input(shape=(X_scores_train.shape[1],))
x = Dense(128, activation='relu')(input_scores)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='linear')(x)

attack_model = tf.keras.Model(inputs=input_scores, outputs=output)
attack_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# コールバック関数の設定
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = attack_model.fit(X_scores_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

loss, mae = attack_model.evaluate(X_scores_test, y_test)
print(f'Test MAE: {mae}')

attack_model.save('small_chess_yatsuka.keras')


# モデルの性能評価
def evaluate_model_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

evaluate_model_performance(attack_model, X_scores_test, y_test)

# 予測結果と実際のラベルを比較
def compare_predictions(model, X_test, y_test, num_samples=10):
    predictions = model.predict(X_test)
    for i in range(num_samples):
        print(f"Sample {i+1}:")
        print(f"Predicted: {predictions[i][0]}, Actual: {y_test[i]}")
        print()

compare_predictions(attack_model, X_scores_test, y_test, num_samples=10)


def load_training_data(file_path='minmax_chess_ver6.json'):
    game_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            game_id = data['game_id']
            if game_id not in game_data:
                game_data[game_id] = {'scores': [], 'turns': []}
            if 'move_score' in data and 'board_score' in data:
                move_score = data.get('move_score', 0)
                board_score = data.get('board_score', 0)
                black_king_threat = data.get('Black_king_threat', 0)
                threatened_pieces = sum(data['threatened_pieces'].values()) if 'threatened_pieces' in data else 0
                can_move_score = data['can_move_score']["white_can_move_score"]
                can_get_score = data['can_move_score']["white_can_get_score"]
                scores = [move_score, board_score, black_king_threat, threatened_pieces, can_move_score, can_get_score]
                game_data[game_id]['scores'].append(scores)
                game_data[game_id]['turns'].append(data['piece'])

    X_scores, y = [], []
    for game_id, game in game_data.items():
        scores = game['scores']
        turns = game['turns']
        for i in range(len(scores) - 1):
            if turns[i].startswith('b'):  # 黒のターンを入力として使用
                current_score = scores[i]
                if i + 1 < len(scores):
                    next_white_king_threat = scores[i + 1][1]  # 次の白のターン終了時の脅威スコア
                    X_scores.append(current_score)
                    y.append(next_white_king_threat)

    return np.array(X_scores), np.array(y)

X_scores, y = load_training_data()

if X_scores.size == 0 or y.size == 0:
    raise ValueError("No data available for training.")

X_scores_train, X_scores_test, y_train, y_test = train_test_split(X_scores, y, test_size=0.2, random_state=42)

input_scores = Input(shape=(X_scores_train.shape[1],))
x = Dense(128, activation='relu')(input_scores)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='linear')(x)

defence_model = tf.keras.Model(inputs=input_scores, outputs=output)
defence_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# コールバック関数の設定
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = defence_model.fit(X_scores_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

loss, mae = defence_model.evaluate(X_scores_test, y_test)
print(f'Test MAE: {mae}')

defence_model.save('small_chess_okitsu.keras')

# モデルの性能評価
evaluate_model_performance(defence_model, X_scores_test, y_test)

# 予測結果と実際のラベルを比較
compare_predictions(defence_model, X_scores_test, y_test, num_samples=10)
