# University-Dissertation

# 📈 Stock Market Prediction using Machine Learning (LSTM vs GRU)

This project investigates the use of deep learning models—specifically **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit)—to predict stock market performance using historical data. It includes technical experimentation, performance comparisons, and a simulated trading strategy.

## 🧠 Project Overview

The goal of the project is to evaluate the predictive performance of LSTM and GRU networks on real-world stock data. The models were trained on stock index data (S&P 500 and Euronext 100) and evaluated using Mean Squared Error (MSE) and a custom trading strategy.

We compare:

- The accuracy of **predicted stock prices** vs. actual prices.
- The profitability of using each model in a simple **trading strategy**.
- Model performance across different datasets.

## 📊 Technologies Used

- Python 3.9
- Keras
- TensorFlow
- NumPy, Pandas
- Matplotlib
- Yahoo Finance (via `pandas_datareader`)

## 🔍 Models Compared

### LSTM (Long Short-Term Memory)

A popular RNN architecture that solves the vanishing gradient problem with gated memory units.

### GRU (Gated Recurrent Unit)

A simplified RNN similar to LSTM but with fewer gates and faster training. The results showed GRU outperformed LSTM in nearly every test.

## 📁 Data Sources

- **S&P 500 Index**
- **Euronext 100 Index**

Data was obtained from **Yahoo Finance** using `pandas_datareader`.

## 🧪 Methodology

1. **Data Preprocessing**:
   - Normalized using `MinMaxScaler`
   - Time-series windowed with 180-day lookbacks
2. **Model Training**:
   - 75% training, 25% testing
   - MSE used as the loss function
   - Epochs: 100, Batch Size: 64
3. **Evaluation**:
   - MSE comparison
   - Graphical evaluation of predictions
   - Profit calculation using a trading strategy with an initial budget of $1000

## 📈 Key Results

### For S&P 500:

| Model | Avg. Profit | Avg. MSE |
| ----- | ----------- | -------- |
| LSTM  | ~$230       | ~15,000  |
| GRU   | ~$240       | ~5,000   |

### For Euronext 100:

| Model | Avg. Profit | Avg. MSE |
| ----- | ----------- | -------- |
| LSTM  | ~$25        | ~1,500   |
| GRU   | ~$28        | ~600     |

✅ **GRU outperformed LSTM** in every metric.

## 💬 Reflections

> "We initially expected LSTM to outperform due to its popularity, but GRU proved more efficient and accurate."

The project highlighted the importance of **evaluating assumptions** and testing multiple models. Future work could involve integrating social media sentiment or real-time news into the predictions.

## 🔮 Future Work

- Compare with more RNN variants (e.g., Bidirectional LSTM)
- Extend to **cryptocurrency prediction**
- Integrate external signals like **news or social media sentiment**
- Explore reinforcement learning for trading decisions

## 📜 Author

**Victor Osisanya**  
University of Kent  
Supervised by Dr. Xiaowei Gu  
Email: vo54@kent.ac.uk

## 🧾 License

This project is for educational and research purposes. No financial advice is provided.
