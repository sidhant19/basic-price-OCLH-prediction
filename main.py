from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
app = Flask(__name__)

def ml_market(ticker, capital1, leverage1):
    print("Loading Data")
    dfm = pd.read_csv('stock_data.csv')
    dfm.dropna(inplace=True)
    ddf = dfm[dfm['ticker'] == ticker].copy()
    df = ddf.copy()

    print("Processing Data")
    df['open'] = df['open'].astype(float)
    df['open1'] = df['open'].shift(-1)
    df['high1'] = df['high'].shift(-1)
    df['low1'] = df['low'].shift(-1)
    df['close1'] = df['close'].shift(-1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['volume_change_pct'] = df['volume'].pct_change() * 100
    df['timestamp_hours'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    df.dropna()

    print("Splitting Data")
    df1 = df.iloc[:int(0.93 * len(df))]
    df2 = df.iloc[int(0.93 * len(df)):]
    df1.dropna()
    df2.dropna()
    features = ['open', 'high',
                'low', 'close', 'volume_change_pct',
                'timestamp_hours']
    target = ['low1', 'high1', 'open1', 'close1']
    X_train = df1[features]
    y_train = df1[target]
    X_test = df2[features]
    y_test = df2[target]

    print("Training Model")
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    print("Predicting Data")
    predictions = model.predict(X_test)

    print("Simulating Trading")

    capital = float(capital1)  # Initial capital in USD
    leverage = float(leverage1)  # Leverage factor
    position = 0  # Initial position (0 for no position, 1 for long, -1 for short)
    cumulative_profit_loss = 0  # Cumulative profit/loss
    cumulative_profit_losses = []

    # Iterate through the common indices of X_test and predictions
    num_samples = min(len(X_test), len(predictions))

    for i in range(num_samples):
        # Get the corresponding row from X_test and the prediction
        row = X_test.iloc[i]
        prediction = predictions[i]
        if prediction[2] < prediction[3] and position != 1:  # Buy signal
            position = 1
            entry_price = y_test.iloc[i][2]
            position_size = capital * leverage / entry_price  # Adjust position size based on entry price
        elif prediction[2] > prediction[3] and position != -1:  # Sell signal
            position = -1
            entry_price = y_test.iloc[i][2]
            position_size = -capital * leverage / entry_price  # Adjust position size based on entry price
        else:  # No signal or already in position
            cumulative_profit_losses.append(cumulative_profit_loss / 100)
            continue

            # Calculate profit/loss for the trade
        exit_price = y_test.iloc[i][3]
        trade_profit_loss = (exit_price - entry_price) * position_size

        # Update capital and cumulative profit/loss
        capital += trade_profit_loss
        cumulative_profit_loss += trade_profit_loss
        cumulative_profit_losses.append(cumulative_profit_loss / 100)
    plt.plot(cumulative_profit_losses)
    plt.xlabel('Time- 1U-5Mins')
    plt.ylabel('P&L %')
    plt.title('P&L')
    plt.grid(True)
    dft=pd.read_csv('res.csv')
    rid=int(dft["reqid"][0])
    plt.savefig(f'static/{rid}_0')


    fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    lb = ['Low', 'High', 'Open', 'Close']
    for i in range(4):
        ax = axes[i]
        ax.plot(range(0, len(y_test)), y_test.iloc[:, i], label='Actual', color='blue')
        ax.plot(range(0, len(y_test)), predictions[:, i], label='Predicted', color='red')

        ax.set_xlabel('Timestamp')
        ax.set_ylabel(f'Target Variable {i + 1}')
        ax.set_title(f'Actual vs Predicted for Target Variable {lb[i]}')
        ax.legend()
    ax.figure.savefig(f'static/{rid}_1')
    dft['reqid'] += 1
    dft.to_csv('res.csv')
    return [rid, cumulative_profit_loss]




@app.route('/', methods=['POST', 'GET'])
def index():
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'JNJ', 'JPM']
    if request.method == 'POST':
        ticker = request.form['ticker']
        capital = request.form['capital']
        leverage = request.form['leverage']
        rid1 = ml_market(ticker, capital, leverage)
        rid = rid1[0]
        datas = [f'static/{rid}_0.png', f'static/{rid}_1.png']
        return render_template('index.html', tickers=tickers, ticker=ticker, capital=capital, leverage=leverage, datas=datas, cd=str(rid1[1]))




    return render_template('index.html', tickers=tickers)

app.run(host="0.0.0.0", port=5012)

