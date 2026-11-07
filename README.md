# Trading

Nghiên cứu & công cụ trading: **Crypto, Chứng khoán, Vàng/Bạc, Binance Futures** — kết hợp **thống kê, thuật toán, Machine Learning & AI**.

> ⚠️ Chỉ phục vụ nghiên cứu/học tập. Không phải lời khuyên đầu tư.

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
