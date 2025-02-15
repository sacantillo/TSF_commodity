import streamlit as st
import yfinance as yf
import pandas as pd
from neuralprophet import NeuralProphet, set_random_seed
import plotly.graph_objects as go

def download_data(ticker,period):
    data = yf.download(ticker, period=period)
    data.reset_index(inplace=True)
    data.columns = data.columns.droplevel(1)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def train_model(df, horizonte):
    set_random_seed(42)
    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_changepoints=50,
        batch_size=64,
        ar_layers=[21, 21],
        n_lags=30,
        n_forecasts=horizonte,
        quantiles=[0.1, 0.9]
    )
    train, val = model.split_df(df, freq='D', valid_p=0.2)
    with st.spinner("Ajustando el modelo de pronóstico, por favor espere..."):
        metrics = model.fit(df, validation_df=val, freq="D", checkpointing=True, early_stopping=True)
    return model, metrics

def forecast_model(model, df, horizon, fecha):
    periods = range(1, horizon+1)
    future = model.make_future_dataframe(df, periods=horizon, n_historic_predictions=None)
    forecasts = model.predict(future)
    fila = forecasts[forecasts['ds'] == pd.to_datetime(fecha) + pd.Timedelta(days=1)].iloc[0]
    data_nueva = {
      'ds': forecasts.loc[forecasts['y'].isnull(),'ds'].tolist(),
      'yhat_10': [fila[f'yhat{i} 10.0%']  for i in periods],
      'yhat':   [fila[f'yhat{i}']      for i in periods],
      'yhat_90': [fila[f'yhat{i} 90.0%']  for i in periods]
    }
    df_nuevo = pd.DataFrame(data_nueva)
    return df_nuevo

def plot_forecast(df, forecast):
    fig = go.Figure()
    # Trazos para cuantiles con azul claro y títulos "Valor mínimo" y "Valor máximo"
    fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_10'],
            mode='lines',
            line=dict(dash='dash', color='rgba(173,216,230,1)'),
            name='Valor mínimo'
        ))
    fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_90'],
            mode='lines',
            line=dict(dash='dash', color='rgba(173,216,230,1)'),
            fill='tonexty',
            fillcolor='rgba(173,216,230,0.2)',
            name='Valor máximo'
        ))
    fig.add_trace(go.Scatter(
            x=df['ds'], y=df['y'], mode='lines', name='Histórico'
        ))
    fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Pronóstico'
        ))
    last_date = forecast['ds'].min()
    fig.add_vline(x=last_date, line_width=2, line_dash="dash", line_color="green")
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Pronóstico con NeuralProphet")
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 3])
    with col1:
        historico = st.slider("Historico (Años)", min_value=1, max_value=11, value=9, step=1)
    with col2:
        horizonte = st.slider("Días a pronosticar", min_value=1, max_value=14, value=7, step=1)
    st.divider()
    data = download_data('SB=F',period=f"{historico}y")[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model, metric = train_model(data, horizonte)

    st.sidebar.header("Métricas de Rendimiento:")
    mae, rmse = metric[['MAE', 'RMSE']].iloc[-1].values

    st.sidebar.metric("MAE", f"{mae:.2f}")
    st.sidebar.metric("RMSE", f"{rmse:.2f}")

    forecast = forecast_model(model, data, horizonte, data['ds'].max())

    st.header("Resultados de Pronóstico:")
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_forecast(data, forecast), use_container_width=True)
    st.dataframe(forecast, use_container_width=True)

if __name__ == "__main__":
    main()