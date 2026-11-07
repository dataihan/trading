# Trading

Nghiên cứu & công cụ trading: **Crypto, Chứng khoán, Vàng/Bạc, Binance Futures** — kết hợp **thống kê, thuật toán, Machine Learning & AI**.

> ⚠️ Chỉ phục vụ nghiên cứu/học tập. Không phải lời khuyên đầu tư.

## 🚀 Kết quả hiện tại
Xem [`results/REPORT.md`](results/REPORT.md). `portfolio_v1` có lời ở cả 3 giai đoạn train / valid / holdout, đã trừ phí và funding (holdout: +13.9%, Sharpe 1.05, max DD -7.1%).

## 🤖 Hệ thống agent tự nghiên cứu
Xem [`program.md`](program.md). Có 4 agent trong `.claude/agents/`: researcher → strategy-builder → backtester → risk-auditor. Thay đổi nào cải thiện kết quả thì commit, không thì reset.

## Cấu trúc

```
markets/                 # Ghi chú theo thị trường
  crypto/                # Spot crypto (BTC, ETH, altcoin...)
  stocks/                # Chứng khoán VN (HOSE/HNX) & quốc tế
  metals/                # Vàng (XAU), Bạc (XAG)
  futures-binance/       # Binance USDⓈ-M / COIN-M futures, funding, leverage
research/
  statistics/            # Phân phối lợi nhuận, volatility, correlation, cointegration
  algorithms/            # Mean reversion, momentum, grid, arbitrage, market making
  machine-learning/      # Feature engineering, XGBoost, LSTM, Transformers
  ai-agents/             # LLM / agentic trading, sentiment analysis
src/trading/
  data/                  # Kết nối API (Binance, yfinance, vnstock...)
  indicators/            # Chỉ báo kỹ thuật (MA, RSI, MACD, ATR...)
  strategies/            # Chiến lược
  backtest/              # Backtesting engine
  ml/                    # Model training / inference
  risk/                  # Position sizing, stop-loss, drawdown control
notebooks/               # Jupyter notebooks
data/                    # raw / processed (không commit dữ liệu lớn)
tests/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 autoresearch/portfolio.py          # tự tải data + backtest
cp .env.example .env   # điền API key (KHÔNG commit .env)
```

## Roadmap

- [ ] Data pipeline: Binance (spot + futures), yfinance, vnstock, XAU/XAG
- [ ] Bộ chỉ báo kỹ thuật cơ bản
- [ ] Backtest engine + metrics (Sharpe, Sortino, Max DD, Win rate)
- [ ] Chiến lược baseline: MA crossover, RSI mean reversion, funding-rate arbitrage
- [ ] ML: dự đoán hướng giá (classification) + walk-forward validation
- [ ] AI: sentiment từ news/social bằng LLM
- [ ] Paper trading trên Binance Testnet
