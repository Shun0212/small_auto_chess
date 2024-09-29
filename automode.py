import smallchess_6
import time
import test
import smallchess_6

def save_execution_count(count, wins, total):
    """実行回数と勝率をexecution_count.txtに保存する"""
    win_rate = wins / total if total > 0 else 0
    with open("execution_count.txt", "w") as file:
        file.write(f"Total execution count: {count}\n")
        file.write(f"Total wins: {wins}\n")
        file.write(f"Win rate: {win_rate:.2%}\n")

def main():
    start_time = time.time()
    execution_count = 0
    wins = 0
    run_time_limit = 24 * 60 * 60  # 24時間（秒換算）
    
    while time.time() - start_time < run_time_limit and execution_count < 500:
        result = smallchess_6.main()
        if result == 'Black wins!':
            wins += 1
        execution_count += 1
        print(f"execution count: {execution_count}, wins: {wins}")

    save_execution_count(execution_count, wins, execution_count)
    print(f"Total execution count: {execution_count}, wins: {wins}")
    print(f"Win rate: {wins/execution_count:.2%}")

if __name__ == "__main__":
    main()
