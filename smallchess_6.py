import pygame
import random
import time
import json
import uuid
import numpy as np
from keras.models import load_model
import copy

# 定数の定義
WIDTH, HEIGHT = 300, 300
ROWS, COLS = 6, 6
SQUARE_SIZE = WIDTH // COLS
CELL_SIZE = SQUARE_SIZE
start_time = None
AIMODE = True
# モデルのロード
if AIMODE:
    attack_model = load_model('small_chess_totsuka.keras')  # 黒用モデルをロード
    defence_model = load_model('small_chess_yata.keras') 

# 色の定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

# ミニマックス法
def minimax(board, depth, maximizing_player, alpha, beta):
    if depth == 0 or check_game_over(board):
        return evaluate_board(board), None

    if maximizing_player:  # 黒側のターン
        max_score = -np.inf
        best_move = None
        for move in get_all_moves(board, "b"):
            new_board = make_move(board, move)
            score, _ = minimax(new_board, depth - 1, False, alpha, beta)
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score, best_move
    else:  # 白側のターン
        min_score = np.inf
        best_move = None
        for move in get_all_moves(board, "w"):
            new_board = make_move(board, move)
            score, _ = minimax(new_board, depth - 1, True, alpha, beta)
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score, best_move

# ボードスコア計算部分
def evaluate_board(board):
    score = 0
    for row in board:
        for piece in row:
            if piece.startswith("b"):
                score += SCORES.get(piece[1], 0)
            elif piece.startswith("w"):
                score -= SCORES.get(piece[1], 0)
    return score

# プレイスコア計算部分
def captured_score(captured):
    score = 0
    if captured.startswith("w"):
        score += SCORES.get(captured[1], 0)
    elif captured.startswith("b"):
        score -= SCORES.get(captured[1], 0)
    return score

# すべての動きを取得
def get_all_moves(board, color):
    moves = []
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col].startswith(color):
                piece_moves = get_valid_moves((row, col), board)
                for move in piece_moves:
                    moves.append(((row, col), move))
    return moves

def evaluate_piece_threat(board, position, color):
    threat_score = 0
    opponent_color = 'b' if color == 'w' else 'w'
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col].startswith(opponent_color):
                if is_valid_move((row, col), position, board):
                    threat_score += SCORES.get(board[row][col][1], 0)
                    if board[row][col][1] == 'K':  # If threatened by King
                        threat_score += 3 if color == 'w' else -3
    return threat_score

def make_move(board, move):
    start, end = move
    new_board = [row[:] for row in board]
    piece = new_board[start[0]][start[1]]
    target = new_board[end[0]][end[1]]
    new_board[end[0]][end[1]] = piece
    new_board[start[0]][start[1]] = '--'
    return new_board

# ミニマックス法を動かす部分
def ai_move_minimax(color, board, depth, prev_movement):
    maximizing_player = color == "b"  # 黒の場合は最大化
    _, best_move = minimax(board, depth, maximizing_player, -np.inf, np.inf)

    # ミニマックス法のループを防ぐための関数
    if best_move in prev_movement:
        best_move = random_move(board, color)[:2]  # 開始位置と終了位置のみ取得

    prev_movement.append(best_move)
    return best_move

# ボード状態を数値ベクトルに変換する関数
def board_to_vector(board):
    piece_map = {
        '--': 0,
        'wP': -1, 'wR': -2, 'wN': -3, 'wB': -4, 'wQ': -5, 'wK': -6,
        'bP': 1, 'bR': 2, 'bN': 3, 'bB': 4, 'bQ': 5, 'bK': 6
    }
    vector = []
    for row in board:
        for cell in row:
            vector.append(piece_map.get(cell, 0))
    return np.array(vector).reshape(1, -1)  # Reshape to (1, 36)

def ai_move(color, board, attack_model, defence_model):
    valid_moves = []
    king_capture_moves = []
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col].startswith(color):
                moves = get_valid_moves((row, col), board)
                for move in moves:
                    if board[move[0]][move[1]] == ('wK' if color == 'b' else 'bK'):
                        king_capture_moves.append(((row, col), move))
                    else:
                        valid_moves.append(((row, col), move))

    # If there are moves that capture the king, choose one of them
    if king_capture_moves:
        best_move = random.choice(king_capture_moves)
        return best_move[0], best_move[1], board[best_move[0][0]], board[best_move[1][0]]

    best_move = None
    best_score = -np.inf

    # First, use the defense model
    for move in valid_moves:
        start, end = move
        piece = board[start[0]][start[1]]
        new_board = make_move(board, (start, end))  # Simulate the move
        move_score = captured_score(board[end[0]][end[1]])
        board_score = evaluate_board(new_board)
        white_king_threat, black_king_threat = evaluate_king_threat(new_board)
        threatened_pieces = get_the_all_threat(new_board)
        sum_threatened_pieces = sum(threatened_pieces.values())
        can_move_score = calculate_can_move_score(new_board)

        # Combine all scores into a single array, now with 6 features
        scores_vector = np.array([move_score, board_score, black_king_threat, sum_threatened_pieces, can_move_score['white_can_move_score'], can_move_score['white_can_get_score']]).reshape(1, -1)

        # Ensure the scores_vector matches the expected shape of the model
        if scores_vector.shape[1] != 6:
            print(f"Skipping move {move} due to shape mismatch")
            continue

        # Pass the inputs as a single array to the defense model
        score = defence_model.predict(scores_vector)[0]

        print(f"Move: {move}, Predicted defense score: {score}")

        if score > best_score:
            best_move = move
            best_score = score

    print("Best move from defense model:", best_move, "with score:", best_score)

    # If the best move's black_king_threat is not severe, use the attack model
    if best_move is not None and best_score >= 2:
        for move in valid_moves:
            start, end = move
            piece = board[start[0]][start[1]]
            new_board = make_move(board, (start, end))  # Simulate the move
            move_score = captured_score(board[end[0]][end[1]])
            board_score = evaluate_board(new_board)
            white_king_threat, black_king_threat = evaluate_king_threat(new_board)
            threatened_pieces = get_the_all_threat(new_board)
            sum_threatened_pieces = sum(threatened_pieces.values())
            can_move_score = calculate_can_move_score(new_board)

            # Combine all scores into a single array, now with 6 features
            scores_vector = np.array([move_score, board_score, white_king_threat, sum_threatened_pieces, can_move_score['black_can_move_score'], can_move_score['black_can_get_score']]).reshape(1, -1)

            # Ensure the scores_vector matches the expected shape of the model
            if scores_vector.shape[1] != 6:
                print(f"Skipping move {move} due to shape mismatch")
                continue

            # Pass the inputs as a single array to the attack model
            score = attack_model.predict(scores_vector)[0]

            print(f"Move: {move}, Predicted attack score: {score}")

            if score > best_score:
                best_move = move
                best_score = score

        print("Best move from attack model:", best_move, "with score:", best_score)

    if best_move is not None:
        return best_move[0], best_move[1], board[best_move[0][0]], board[best_move[1][0]]
    else:
        return None, None, None, None

def calculate_can_move_score(board):
    """計算動けるマスのスコアを計算"""
    move_score = {
        "black_can_move_score": 0,
        "white_can_move_score": 0,
        "black_can_get_score": 0,
        "white_can_get_score": 0
    }
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece != "--":
                moves = get_valid_moves((row, col), board)
                for move in moves:
                    if piece.startswith("b"):
                        move_score["black_can_move_score"] += 1
                        if board[move[0]][move[1]] != "--":
                            move_score["black_can_get_score"] += SCORES.get(board[move[0]][move[1]][1], 0)
                    elif piece.startswith("w"):
                        move_score["white_can_move_score"] -= 1
                        if board[move[0]][move[1]] != "--":
                            move_score["white_can_get_score"] -= SCORES.get(board[move[0]][move[1]][1], 0)

    return move_score

def is_under_threat(board, piece_pos, is_white):
    piece_values = {
        'P': 1, 'R': 5, 'N': 3, 'B': 3, 'Q': 9, 'K': 3
    }
    threat_score = 0
    opponent_color = 'w' if not is_white else 'b'
    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            if piece.startswith(opponent_color):
                if is_valid_move((i, j), piece_pos, board):
                    threat_score += piece_values[piece[1]]
    return threat_score

def get_the_all_threat(board):
    piece_values = {
        'P': 1, 'R': 5, 'N': 3, 'B': 3, 'Q': 9, 'K': 10
    }
    threat_score = {
        "bR": 0,
        "bN": 0,
        "bB": 0,
        "bQ": 0,
        "bK": 0,
    }

    # 初期駒の数を設定
    piece_counts = {
        "bR": 2,
        "bN": 2,
        "bB": 2,
        "bQ": 1,
        "bK": 1,
    }

    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            if piece in threat_score:
                threat_value = is_under_threat(board, (i, j), piece.startswith("w"))
                if piece.startswith("b"):
                    threat_value = -threat_value
                threat_score[piece] += threat_value * piece_values[piece[1]]
                piece_counts[piece] -= 1

    # 存在しない駒に対してスコアをマイナス
    for piece, count in piece_counts.items():
        if count > 0:
            threat_score[piece] -= count * piece_values[piece[1]]

    return threat_score

def save_autoplay_move(piece, start, end, target, move_score, board_score, white_king_threat, black_king_threat, can_move_score, board, game_id, level, promotion=None, promotion_score=0,turn =0):
    """移動をリストに追加する"""
    move = {
        "game_id": game_id,
        "start_time": start_time,
        "turn":turn,
        "piece": piece,
        "board_score": board_score,
        "move_score": move_score,
        "White_king_threat": white_king_threat, 
        "Black_king_threat": black_king_threat,
        "can_move_score": can_move_score,
        "level": level,
        "start": start,
        "end": end,
        "captured": target if target != "--" else None,
        "threatened_pieces": get_the_all_threat(board),
        "Human": False,
        "promotion": promotion,
        "promotion_score": promotion_score
    }
    if(AIMODE==False):
        with open("minmax_chess_ver6.json", "a") as file:
            file.write(json.dumps(move) + "\n")
    else:
        with open("level5_chess_ver6.json", "a") as file:
            file.write(json.dumps(move) + "\n")

def evaluate_king_threat(board):
    white_king_pos = None
    black_king_pos = None

    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            if piece == 'wK':
                white_king_pos = (i, j)
            elif piece == 'bK':
                black_king_pos = (i, j)

    white_king_threat = is_under_threat(board, white_king_pos, True) if white_king_pos else 0
    black_king_threat = -is_under_threat(board, black_king_pos, False) if black_king_pos else 0

    return white_king_threat, black_king_threat

def random_move(board, color):
    """ランダムに有効な移動を取得する"""
    valid_moves = []
    capture_moves = []
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col].startswith(color):
                moves = get_valid_moves((row, col), board)
                for move in moves:
                    if board[move[0]][move[1]] != "--":
                        capture_moves.append(((row, col), move))
                    else:
                        valid_moves.append(((row, col), move))
    if capture_moves:
        start, end = random.choice(capture_moves)
    elif valid_moves:
        start, end = random.choice(valid_moves)
    piece = board[start[0]][start[1]]
    target = board[end[0]][end[1]]
    return start, end, piece, target

# 得点の定義
SCORES = {
    "P": 1,
    "R": 5,
    "B": 3,
    "N": 3,
    "Q": 9,
    "K": 10  # キングはゲーム終了条件
}

# フォントの初期化
def initialize_fonts():
    global FONT
    FONT = pygame.font.SysFont('arial', 18)

# ピースのイメージのロード
def load_piece(piece_name):
    piece = pygame.image.load(f'asset/{piece_name}.png').convert_alpha()
    return pygame.transform.scale(piece, (CELL_SIZE - 10, CELL_SIZE - 10))

# ピースをロード
def load_pieces():
    return {
        "bP": load_piece("BPawn"),
        "bR": load_piece("BRook"),
        "bN": load_piece("BKnight"),
        "bB": load_piece("BBishop"),
        "bQ": load_piece("BQueen"),
        "bK": load_piece("BKing"),
        "wP": load_piece("WPawn"),
        "wR": load_piece("WRook"),
        "wN": load_piece("WKnight"),
        "wB": load_piece("WBishop"),
        "wQ": load_piece("WQueen"),
        "wK": load_piece("WKing")
    }

def draw_board(win):
    """チェスボードを描画し、座標を表示する"""
    win.fill(WHITE)
    for row in range(ROWS):
        for col in range(COLS):
            color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            text = FONT.render(f'{row}{col}', True, TEXT_COLOR)
            win.blit(text, (col * SQUARE_SIZE + SQUARE_SIZE - 20, row * SQUARE_SIZE + SQUARE_SIZE - 20))

def draw_pieces(win, board, pieces):
    """駒を描画する"""
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece != "--":
                win.blit(pieces[piece], (col * SQUARE_SIZE + 5, row * SQUARE_SIZE + 5))

def is_valid_move(start, end, board):
    """駒の移動が有効かをチェックする関数"""
    s_row, s_col = start
    e_row, e_col = end
    piece = board[s_row][s_col]
    target = board[e_row][e_col]

    if target != "--" and target[0] == piece[0]:
        return False

    if piece[1] == "P":  # ポーンの動き
        if piece[0] == "w":  # 白のポーン
            if s_col == e_col and target == "--" and (e_row == s_row - 1 or (s_row == 5 and e_row == 4)):
                return True
            if abs(s_col - e_col) == 1 and e_row == s_row - 1 and target.startswith("b"):
                return True
        elif piece[0] == "b":  # 黒のポーン
            if s_col == e_col and target == "--" and (e_row == s_row + 1 or (s_row == 0 and e_row == 1)):
                return True
            if abs(s_col - e_col) == 1 and e_row == s_row + 1 and target.startswith("w"):
                return True

    elif piece[1] == "R":  # ルークの動き
        if s_row == e_row or s_col == e_col:
            if all(board[r][c] == "--" for r, c in get_path(start, end)):
                return True

    elif piece[1] == "N":  # ナイトの動き
        if (abs(s_row - e_row), abs(s_col - e_col)) in [(2, 1), (1, 2)]:
            return True

    elif piece[1] == "B":  # ビショップの動き
        if abs(s_row - e_row) == abs(s_col - e_col):
            if all(board[r][c] == "--" for r, c in get_path(start, end)):
                return True

    elif piece[1] == "Q":  # クイーンの動き
        if s_row == e_row or s_col == e_col or abs(s_row - e_row) == abs(s_col - e_col):
            if all(board[r][c] == "--" for r, c in get_path(start, end)):
                return True

    elif piece[1] == "K":  # キングの動き
        if max(abs(s_row - e_row), abs(s_col - e_col)) == 1:
            return True

    return False

def get_path(start, end):
    """スタートからエンドまでのパスを取得する"""
    s_row, s_col = start
    e_row, e_col = end
    path = []

    if s_row == e_row:
        step = 1 if s_col < e_col else -1
        for col in range(s_col + step, e_col, step):
            path.append((s_row, col))
    elif s_col == e_col:
        step = 1 if s_row < e_row else -1
        for row in range(s_row + step, e_row, step):
            path.append((row, s_col))
    elif abs(s_row - e_row) == abs(s_col - e_col):
        row_step = 1 if s_row < e_row else -1
        col_step = 1 if s_col < e_col else -1
        for row, col in zip(range(s_row + row_step, e_row, row_step), range(s_col + col_step, e_col, col_step)):
            path.append((row, col))

    return path

def move_piece(board, start, end):
    """駒を移動する"""
    s_row, s_col = start
    e_row, e_col = end
    target = board[e_row][e_col]
    piece = board[s_row][s_col]
    board[e_row][e_col] = piece
    board[s_row][s_col] = "--"
    
    promotion = None
    promotion_score = 0
    
    # ポーンのプロモーション
    if piece[1] == "P" and (e_row == 0 or e_row == 5):
        promotion_choices = ["Q", "R", "B", "N"]
        promotion_choice = random.choice(promotion_choices)
        board[e_row][e_col] = piece[0] + promotion_choice
        promotion = promotion_choice
        promotion_score = SCORES[promotion_choice] * (1 if piece[0] == 'b' else -1)
    
    return target, promotion, promotion_score

def get_valid_moves(selected, board):
    """選択された駒の有効な移動先を取得する"""
    valid_moves = []
    for row in range(ROWS):
        for col in range(COLS):
            if is_valid_move(selected, (row, col), board):
                valid_moves.append((row, col))
    return valid_moves

def draw_valid_moves(win, valid_moves):
    """有効な移動先を描画する"""
    for move in valid_moves:
        row, col = move
        pygame.draw.rect(win, RED, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

def start_draw(win, row, col, BLUE):
    """最初の位置を描画する"""
    pygame.draw.rect(win, BLUE, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

def king_draw(win, row, col, color):
    """KINGの位置を描画する"""
    pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 10)

def check_game_over(board):
    """ゲームオーバーをチェックする"""
    kings = sum(row.count("wK") for row in board) + sum(row.count("bK") for row in board)
    return kings < 2

def is_in_check(board, color):
    """チェックされているかを確認する"""
    king_position = None
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == f"{color}K":
                king_position = (row, col)
                break
        if king_position:
            break
    
    if not king_position:
        return False

    opponent_color = "b" if color == "w" else "w"
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col].startswith(opponent_color):
                if is_valid_move((row, col), king_position, board):
                    return True
    return False

def is_checkmate(board, color):
    """チェックメイトを確認する"""
    if not is_in_check(board, color):
        return False

    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col].startswith(color):
                valid_moves = get_valid_moves((row, col), board)
                for move in valid_moves:
                    new_board = make_move(board, ((row, col), move))
                    if not is_in_check(new_board, color):
                        return False
    return True

def save_result(game_id, result, board_score, turn):
    """結果とスコアをresult.jsonに保存する"""
    result_data = {
        "game_id": game_id,
        "result": result,
        "final_board_score": board_score,
        "timestamp": start_time,
        "turn": turn
    }
    if(AIMODE==False):
        with open("minmax_chess_ver6.json", "a") as file:
            file.write(json.dumps(result_data) + "\n")
    else:
        with open("level5_chess_ver6.json", "a") as file:
            file.write(json.dumps(result_data) + "\n")

def get_threatened_pieces_score(board):
    threatened_score = 0
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece != "--" and piece[1] != 'B':
                threatened_score += 10 * evaluate_piece_threat(board, (row, col), piece[0])
    return threatened_score

def update_board_score(target, score):
    """スコアを更新する"""
    if target != "--":
        if target[0] == "w":
            score += SCORES[target[1]]
        else:
            score -= SCORES[target[1]]
    return score

def main():
    # Pygameの初期化
    global start_time
    start_time = time.time()
    pygame.init()
    initialize_fonts()
    board_score = 0
    move_score = 0

    # ウィンドウの作成
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess")

    # ピースのロード
    PIECES = load_pieces()

    # 駒の初期配置
    board = [
        ["bR", "bB", "bQ", "bK", "bN", "bR"],
        ["bP", "bP", "bP", "bP", "bP", "bP"],
        ["--", "--", "--", "--", "--", "--"],
        ["--", "--", "--", "--", "--", "--"],
        ["wP", "wP", "wP", "wP", "wP", "wP"],
        ["wR", "wB", "wQ", "wK", "wN", "wR"]
    ]

    selected = None
    turn = "w"  # 白が先手
    valid_moves = []
    game_over = False
    input_str = ""
    game_id = str(uuid.uuid4())  # 一意のゲーム識別子
    black_movement = []
    white_movement = []
    turnnum = 0
    run = True
    moves = []
    white_king_threat = 0
    black_king_threat = 0
    target = None  # Initialize target to avoid reference before assignment

    while run:
        before_board = copy.deepcopy(board)
        turnnum += 1
        level = random.randint(4, 4)

        # 黒の手番
        if turn == "b" and not game_over:
            if AIMODE:
                start, end = ai_move("b", board, attack_model,defence_model)[:2]
            else:
                 start, end = ai_move_minimax("b", board, 4, black_movement)
            if start is not None and end is not None and is_valid_move(start, end, board):
                selected = start
                piece = board[start[0]][start[1]]  # Assign piece here
                valid_moves = get_valid_moves(selected, board)
                target, promotion, promotion_score = move_piece(board, start, end)
                board_score = evaluate_board(board)
                move_score = captured_score(target) + promotion_score  # プロモーションによる点数を追加
                white_king_threat, black_king_threat = evaluate_king_threat(board)
                can_move_score = calculate_can_move_score(board)
                save_autoplay_move(piece, start, end, target, move_score, board_score, white_king_threat, black_king_threat, can_move_score, board, game_id, level, promotion, promotion_score,turnnum)
                if target == "wK":
                    result = "Black wins!"
                    game_over = True
                turn = "w"

        # 白の手番
        elif turn == "w" and not game_over:
            if AIMODE==False:
                start, end = ai_move_minimax("w", board, level, white_movement)
            else:
                start, end = ai_move_minimax("w", board, level, white_movement)
            if start is not None and end is not None and is_valid_move(start, end, board):
                selected = start
                valid_moves = get_valid_moves(selected, board)
                piece = board[start[0]][start[1]]  # Assign piece here
                target, promotion, promotion_score = move_piece(board, start, end)
                board_score = evaluate_board(board)
                move_score = captured_score(target) + promotion_score  # プロモーションによる点数を追加
                white_king_threat, black_king_threat = evaluate_king_threat(board)
                can_move_score = calculate_can_move_score(board)
                save_autoplay_move(piece, start, end, target, move_score, board_score, white_king_threat, black_king_threat, can_move_score, board, game_id, level, promotion, promotion_score,turnnum)
                if target == "bK":
                    result = "White wins!"
                    game_over = True
                turn = "b"
        draw_board(WIN)
        draw_pieces(WIN, board, PIECES)
        if selected:
            start_draw(WIN, selected[0], selected[1], BLUE)
        if valid_moves:
            draw_valid_moves(WIN, valid_moves)
        if target and target[1] == "K":
            king_draw(WIN, end[0], end[1], WHITE if target == "wK" else BLACK)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if turnnum > 300:
            result = None

        if game_over or turnnum > 300:
            run = False
    save_result(game_id, result, board_score, turnnum)
    print(result)
    pygame.quit()
    return result

if __name__ == "__main__":
    main()
