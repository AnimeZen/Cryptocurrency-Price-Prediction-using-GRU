import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create GRU model
class GRUNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(GRUNet, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    # GRU
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True) # if batch_first = True then x should be [batch_size, seq_len, num_feature]
    self.output_layer = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

    out, _ = self.gru(x, hidden_state)
    # out: batch_size, seq_len, hidden_size
    out = self.output_layer(out[:, -1, :])
    return out

def load_data(tickerSymbol):
    # get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)
    # get the historical prices for this ticker
    tickerDf = tickerData.history(start='2021-1-1', end='2023-6-29').reset_index()

    # Take only the Close price
    new_df = tickerDf[['Close']]

    return new_df

st.markdown("<h1 style='text-align: center;'>Cryptocurrency Price Prediction<br>using Gated Recurrent Unit</h1>", unsafe_allow_html=True)

# Define available ticker symbols and their descriptions
ticker_symbols = {
    'Bitcoin (BTC)': {'ticker': 'BTC-USD', 'identifier': 'Bitcoin'},
    'Ethereum (ETH)': {'ticker': 'ETH-USD', 'identifier': 'Ethereum'},
    'Cardano (ADA)': {'ticker': 'ADA-USD', 'identifier': 'Cardano'},
    'Dogecoin (DOGE)': {'ticker': 'DOGE-USD', 'identifier': 'Dogecoin'},
    'Solana (SOL)': {'ticker': 'SOL-USD', 'identifier': 'Solana'},
    'Polygon (MATIC)': {'ticker': 'MATIC-USD', 'identifier': 'Polygon'},
    'TRON (TRX)': {'ticker': 'TRX-USD', 'identifier': 'TRON'},
    'Litecoin (LTC)': {'ticker': 'LTC-USD', 'identifier': 'Litecoin'},
    'Polkadot (DOT)': {'ticker': 'DOT-USD', 'identifier': 'Polkadot'},
    'Avalanche (AVAX)': {'ticker': 'AVAX-USD', 'identifier': 'Avalanche'}
}

# User input for ticker symbol and display type
tickerSymbol = st.selectbox("Select the cryptocurrency:", list(ticker_symbols.keys()))
display_type = st.selectbox("Select the display type:", ['Line Chart', 'Dataframe'])

if tickerSymbol:
    # Get the corresponding ticker symbol
    selected_ticker = ticker_symbols[tickerSymbol]['ticker']

    # Get data on the chosen ticker
    tickerData = yf.Ticker(selected_ticker)

    # Get the historical prices for this ticker
    start_date = '2021-01-01'
    end_date = '2023-06-29'

    tickerDf = tickerData.history(start=start_date, end=end_date).reset_index()
    tickerDf.set_index('Date', inplace=True)  # Set 'Date' column as the index

    if display_type == 'Line Chart':
        st.subheader(tickerSymbol + ' Data trained from 2021 to June 28, 2023')
        st.line_chart(tickerDf[['Close']])
    elif display_type == 'Dataframe':
        st.subheader(tickerSymbol + ' Data trained from 2021 to June 28, 2023')
        dataframe = tickerDf[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        dataframe.index = dataframe.index.strftime('%Y-%m-%d')
        st.write(dataframe)
else:
    st.write("Please select a cryptocurrency.")

# Define the sidebar options
days_options = [15, 30, 45, 60]
hidden_size_options = [32, 64, 128]

# Add the sidebar header
st.sidebar.markdown("## Select Parameters")

# Add the sidebars as sliders
days = st.sidebar.slider("Days", min_value=15, max_value=60, value=30, step=15)
hidden_size = st.sidebar.selectbox("Hidden Size", hidden_size_options, index=1)

# Get the corresponding file locations based on the selected options
file_location = f"file/gru{days}/{hidden_size}"

# Get the identifier
if tickerSymbol:
    ticker_info = ticker_symbols[tickerSymbol]
    identifier = ticker_info['identifier']

    st.write(f"""
    ## Model Evaluation
    Trained with a window size of {days} days and hidden_size of {hidden_size}
    """)
    prediction_file = f"{file_location}/{identifier}_predictions.png"

    # Load and display the prediction file if it exists
    prediction_file_path = Path(prediction_file)
    if prediction_file_path.exists():
        st.image(str(prediction_file_path))  # Convert the path to string
    else:
        st.write("No prediction file found.")

    # Load the GRU model
    model_file = f"{file_location}/models/{identifier}_model.pth"
    model_file_path = Path(model_file)
    
    # Hyperparameters
    window_size = int(days)
    batch_size = 32
    input_size = 1
    hidden_size = int(hidden_size)
    num_layers = 1
    output_size = 1

    if model_file_path.exists():
        # Load the model
        torch.manual_seed(42)
        model = GRUNet(input_size, hidden_size, num_layers, output_size).to(device)
        model.load_state_dict(torch.load(model_file_path, map_location='cpu'))
        model.eval()

        new_df = load_data(ticker_symbols[tickerSymbol]['ticker'])
        
        scaler = StandardScaler()

        pred_train_size = int(len(new_df) * 0.8)
        pred_train_data = new_df[:pred_train_size].copy()
        pred_train_data = scaler.fit_transform(pred_train_data)

        # Get data on the chosen ticker
        tickerData = yf.Ticker(selected_ticker)

        # Get the historical prices for this ticker
        start_date = '2018-01-01'
        end_date = datetime.today().strftime('%Y-%m-%d')  # Get current datex

        tickerDf = tickerData.history(start=start_date, end=end_date).reset_index()
        tickerDf['Date'] = tickerDf['Date'].dt.date  # Extract date only from the datetime columns
        tickerDf.set_index('Date', inplace=True)  # Set 'Date' column as the index
        
        tickerDf = tickerDf[['Close']]

        # Create empty lists to store the original and predicted values
        dates = []
        original_values = []
        predicted_values = []

        # Iterate over the range of days from 1 to 30
        for day in range(1, window_size + 1):
            # Get the last 30 days of closing prices from the new_df dataframe
            last_window_size_days = tickerDf[-(window_size + day):-(day)]
            # Normalize the data
            last_window_size_days = scaler.transform(last_window_size_days[['Close']])
            # Convert into tensor
            last_window_size_days = torch.from_numpy(last_window_size_days).float().to(device)

            # Use the last 30 days to predict the next day's closing price
            model.eval()
            with torch.inference_mode():
                pred = model(last_window_size_days.unsqueeze(0))

            # Denormalize the predicted price
            pred = scaler.inverse_transform(pred.cpu().numpy())

            # Append the date, original, and predicted values to their respective lists
            dates.append(tickerDf.index[-day].strftime('%Y-%m-%d'))
            original_values.append(tickerDf.iloc[-day]['Close'])
            predicted_values.append(pred[0][0])

        # Reverse the order of the lists to match the chronological order
        dates = dates[::-1]
        original_values = original_values[::-1]
        predicted_values = predicted_values[::-1]

        # Create a dataframe of original and predicted values
        df = pd.DataFrame({'Date': dates,
                        'Original': original_values,
                        'Predicted': predicted_values})

        # Use the last 30 days to predict the 31st day's closing price
        last_window_size_days = tickerDf[-window_size:]
        # Normalize the data
        last_window_size_days = scaler.transform(last_window_size_days[['Close']])
        # Convert into tensor
        last_window_size_days = torch.from_numpy(last_window_size_days).float().to(device)

        # Use the last 30 days to predict the next day's closing price
        model.eval()
        with torch.inference_mode():
            pred = model(last_window_size_days.unsqueeze(0))

        # Denormalize the predicted price
        pred = scaler.inverse_transform(pred.cpu().numpy())

        # Get the date of the next day
        next_day = tickerDf.index[-1] + timedelta(days=1)
        next_day_formatted = next_day.strftime('%B %d, %Y')

        st.write(f"""
        ### Prediction
        """)
        display_type_prediction = st.selectbox("Select the display type:", ['Line Chart', 'Dataframe'], key='display_type')

        if display_type_prediction == 'Line Chart':
            # Plotting the original and predicted values
            fig, ax = plt.subplots()
            ax.plot(dates, original_values, label='Original')
            ax.plot(dates, predicted_values, label='Predicted')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'Comparison of original and predicted prices for the last {days} days')
            ax.legend()
            plt.xticks(rotation=90)
            st.pyplot(fig)
        elif display_type_prediction == 'Dataframe':
            st.write(df)

        # Display the predicted price with the corresponding date
        st.write(f"##### Predicted price for {next_day_formatted}: {pred[0][0]:.3f} <span style='font-size:small'>USD</span>", unsafe_allow_html=True)

    else:
        st.write("No GRU model file found.")