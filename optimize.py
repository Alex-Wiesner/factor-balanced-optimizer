from tools.optimizer import solve_weights
import sys


tickers = sys.argv[1:]
print(solve_weights(tickers))
