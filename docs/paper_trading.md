# Paper trading `portfolio_v1`

## 1. Cài đặt (khoảng 15 phút)

**Chuẩn bị key (chỉ dùng tài khoản thử, không dùng tiền thật):**
- Binance Futures testnet: https://testnet.binancefuture.com → đăng nhập → API Key.
  Nếu testnet đã được thay bằng *Demo Trading* thì lấy key trong Demo Trading và đặt
  `BINANCE_FAPI_BASE=https://demo-fapi.binance.com`.
- Alpaca paper (cho QQQ): https://alpaca.markets → Paper Trading → API Keys.

**Chạy trên server (EC2 hoặc máy ở nhà, chạy Ubuntu):**
```bash
git clone https://github.com/dataihan/trading && cd trading
cp .env.example .env && nano .env      # điền key
sudo bash deploy/install.sh
```
Script sẽ cài 4 lịch chạy (timer):

| timer | khi nào | làm gì |
|---|---|---|
| `trading-crypto` | 00:01 UTC mỗi ngày | tính tín hiệu trend → đặt lệnh trên Binance testnet |
| `trading-qqq` | 15:45 giờ New York, thứ 2–6 | tính IBS và turn-of-month → đặt lệnh khớp giá đóng cửa (MOC) trên Alpaca paper |
| `trading-collector` | mỗi giờ | lưu funding, open interest, order book, tỷ lệ long/short (dữ liệu cho round 3) |
| `trading-equity` | 00:30 UTC mỗi ngày | lưu số dư tài khoản để so với backtest |

⚠️ Binance trả lỗi **HTTP 451** khi truy cập từ khu vực bị hạn chế (ví dụ server đặt ở Mỹ).
Bước kiểm tra kết nối trong `install.sh` sẽ báo lỗi này. Khi đó nên chạy bot từ máy hoặc server ở nơi
Binance cho phép (ví dụ máy ở Việt Nam). Xem điều khoản của Binance trước khi đổi region.

### Chạy trên Claude cloud (không cần server)
Key testnet `testnet.binancefuture.com` gọi được từ cloud. Dữ liệu giá thật lấy từ `data.binance.vision`
(file theo tháng + theo ngày, trễ khoảng 1 ngày; file của 1 ngày có lúc khoảng 09:00 UTC hôm sau).
- Biến môi trường của cloud env: `BINANCE_API_KEY`, `BINANCE_PRIVATE_KEY` (Ed25519, 1 dòng có `\n`).
- Hằng ngày (tác vụ định kỳ, khoảng 10:23 UTC): `bash deploy/cloud_run.sh`
- Mỗi thứ 2: `bash deploy/cloud_run.sh report`
- Log (`ledger.jsonl`, `equity.jsonl`) lưu trên branch `paper-log`, vì container cloud bị xoá sau mỗi lần chạy.
- Chỉ chạy phần crypto (`strategies/portfolio_v1_crypto.json`, giữ nguyên tỷ trọng của v1, tổng 50%)
  cho tới khi có key Alpaca. Collector (order book, OI) không chạy được trên cloud vì `fapi.binance.com` bị chặn.

**Lệnh hay dùng:**
```bash
PYTHONPATH=src python3 -m trading.live.runner crypto --dry-run   # xem tín hiệu, không đặt lệnh
PYTHONPATH=src python3 -m trading.live.report                    # so sánh live với backtest
journalctl -u 'trading-*' --since today                         # xem log
```

**An toàn:**
- Mặc định bot chỉ đặt lệnh trên testnet và paper.
- Muốn chạy tiền thật phải tự đổi base URL **và** đặt `ALLOW_LIVE=yes`. Không có bước nào làm việc này tự động.
- Leverage mặc định là 2. Bot chỉ dùng phần vốn `PAPER_CAPITAL_USD` (mặc định 10.000 USD).

---

## 2. Nếu paper trading cho kết quả không tốt: làm gì?

**Ngưỡng đặt trước khi chạy, không sửa sau khi đã có kết quả** (đã code trong `trading/live/report.py`):

| quy tắc | ngưỡng |
|---|---|
| Thời gian tối thiểu trước khi đánh giá | **90 ngày.** v1 chỉ khoảng 6–10 lệnh/tháng, nên 1–2 tháng chưa nói lên gì |
| Kill switch | drawdown thực tế **≤ -15%** (gấp 1,5 lần mức tệ nhất trong backtest là -10,5%) → dừng bot ngay |
| Chênh lệch khi khớp lệnh | live kém backtest **cùng giai đoạn** quá **5%/năm** |

Chạy `python -m trading.live.report`. Kết quả sẽ rơi vào 1 trong 4 trường hợp:

### A. Live kém backtest cùng giai đoạn → lỗi khi khớp lệnh, chiến lược vẫn có thể đúng
Nguyên nhân thường gặp: slippage, đặt lệnh trễ, funding cao hơn giả định, làm tròn khối lượng, lệnh lỗi.
Cách xử lý (không đổi chiến lược):
1. Đọc `ledger.jsonl`: so `signal_close` với `avg_price` để đo slippage thật.
2. Chuyển sang lệnh limit/maker (phí 0.02% thay vì 0.05%), hoặc chia nhỏ lệnh theo thời gian (TWAP).
3. Đưa slippage và funding thật vào lại backtest, rồi kiểm tra v1 còn có lời không.

### B. Live bám sát backtest, nhưng cả hai cùng đi ngang hoặc lỗ → giai đoạn thị trường xấu cho chiến lược
Chiến lược trend có những giai đoạn đi ngang hoặc lỗ kéo dài; năm 2022 backtest cũng -1,7%.
1. Nếu drawdown chưa chạm -15% thì **tiếp tục chạy, không sửa gì.** Đây là chỗ hay sai nhất: sửa
   chiến lược sau vài tuần xui là overfit vào nhiễu.
2. Kiểm tra mức lỗ có nằm trong phân phối lịch sử không: lấy các đoạn 90 ngày trong backtest
   (rolling 90 ngày) xem bao nhiêu % trường hợp tệ bằng hoặc tệ hơn hiện tại.
   Nếu trên 10% số đoạn tệ như vậy thì đây là chuyện bình thường.

### C. Chạm kill switch (DD ≤ -15%) → dừng bot, quay lại nghiên cứu
1. `sudo systemctl stop 'trading-*.timer'`
2. Risk-auditor viết phân tích: phần nào gây lỗ, thị trường đang ở trạng thái nào, giả định nào sai.
3. Chạy round 3 với dữ liệu mới mà collector đã thu thập (order book, OI, long/short).
   Chiến lược mới phải **qua lại toàn bộ quy trình** train → valid → holdout mới → paper 90 ngày.
   Không được lấy lại v1, sửa vài chỗ rồi chạy tiếp.

### D. Một phần tốt, một phần kém
`ledger.jsonl` ghi tín hiệu và lệnh theo từng phần. Chạy backtest riêng từng phần trên cùng giai đoạn (`autoresearch/portfolio.py` in số liệu theo từng phần) để tìm phần kém. Nếu chỉ 1 phần kém (ví dụ SOL) thì chỉ phần đó bị
xét lại. Các phần khác chạy tiếp. Tỷ trọng mới phải được chọn trên train/valid, không dựa vào kết quả paper.

**Nguyên tắc chung:** paper trading là để kiểm tra chiến lược, không phải để tối ưu thêm.
Mỗi thay đổi đều phải tạo phiên bản mới (v2, v3…) và đi lại từ đầu quy trình.

## 3. Khi nào được chạy tiền thật?
Chỉ khi đủ **tất cả** điều kiện sau, và Han tự quyết định:
- ≥ 90 ngày paper trading (nên 6 tháng)
- live bám backtest với chênh lệch ≥ -5%/năm
- không chạm kill switch
- bắt đầu bằng số vốn nhỏ mà mất hết cũng không sao

## 4. Paper trading sleeve carry funding (từ 2026-11-09, thuộc portfolio v2b)
- Code: `src/trading/live/carry.py`. Chạy hằng ngày: `bash deploy/cloud_run.sh carry`; report: `bash deploy/cloud_run.sh carry-report`.
- Chiến lược: 27 coin (bỏ EOS vì testnet không có), mỗi thứ 2 xếp hạng theo tổng funding 7 ngày:
  bán 20% coin funding cao nhất, mua 20% thấp nhất, tổng 50% mỗi bên.
- **Đánh giá bằng shadow PnL** (giá thật + funding thật từ data.binance.vision). Funding 7 ngày gần nhất được
  ước từ premium index (trễ 1 ngày); report tính lại bằng funding thật khi file tháng có.
  Lãi/lỗ trên testnet KHÔNG dùng để đánh giá vì funding testnet là giả.
- Lệnh testnet (`CARRY_CAPITAL_USD`, mặc định 2000 gross) chỉ để kiểm khớp lệnh. Không đặt lệnh BTC/ETH/SOL/BNB
  vì trùng với bot v1. Snapshot số dư ghi `binance_carry_income` và report v1 trừ phần này ra.
- Ngưỡng: như v1, đánh giá sau ≥ 90 ngày; backtest valid Sharpe 0.98, holdout 1.50.
