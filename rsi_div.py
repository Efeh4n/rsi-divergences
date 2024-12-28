import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import datetime
def settings():
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

settings()
bist_100 = [
    'AEFES.IS', 'AGHOL.IS', 'AKBNK.IS', 'AKCNS.IS', 'AKFGY.IS', 'AKFYE.IS', 
    'AKSA.IS', 'AKSEN.IS', 'ALARK.IS', 'ALBRK.IS', 'ALFAS.IS', 'ARCLK.IS', 
    'ASELS.IS', 'ASTOR.IS', 'BERA.IS', 'BIENY.IS', 'BIMAS.IS', 'BRSAN.IS', 
    'BRYAT.IS', 'BUCIM.IS', 'CANTE.IS', 'CCOLA.IS', 'CIMSA.IS', 'CWENE.IS', 
    'DOHOL.IS', 'DOAS.IS', 'ECILC.IS', 'ECZYT.IS', 'EGEEN.IS', 'EKGYO.IS', 
    'ENJSA.IS', 'ENKAI.IS', 'EREGL.IS', 'EUPWR.IS', 'EUREN.IS', 'FROTO.IS', 
    'GARAN.IS', 'GENIL.IS', 'GESAN.IS', 'GLYHO.IS', 'GUBRF.IS', 'GWIND.IS', 
    'HALKB.IS', 'HEKTS.IS', 'IMASM.IS', 'IPEKE.IS', 'ISCTR.IS', 'ISDMR.IS', 
    'ISGYO.IS', 'ISMEN.IS', 'IZDMC.IS', 'KARSN.IS', 'KAYSE.IS', 'KCAER.IS', 
    'KCHOL.IS', 'KMPUR.IS', 'KONTR.IS', 'KONYA.IS', 'KORDS.IS', 'KOZAA.IS', 
    'KOZAL.IS', 'KRDMD.IS', 'KZBGY.IS', 'MAVI.IS', 'MGROS.IS', 'MIATK.IS', 
    'ODAS.IS', 'OTKAR.IS', 'OYAKC.IS', 'PENTA.IS', 'PETKM.IS', 'PGSUS.IS', 
    'PSGYO.IS', 'QUAGR.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'SKBNK.IS', 
    'SMRTG.IS', 'SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 
    'TOASO.IS', 'TSKB.IS', 'TTKOM.IS', 'TTRAK.IS', 'TUPRS.IS', 'ULKER.IS', 
    'VAKBN.IS', 'VESBE.IS', 'VESTL.IS', 'YKBNK.IS', 'YYLGD.IS', 'ZOREN.IS'
]

def rsi_div(ticker, start_date=None, end_date=None, rsi_period=14, atr_period=14, order=2):

    if end_date is None:
        end_date = datetime.date.today()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=365*5)

    df = yf.download(tickers=ticker, start=start_date, end=end_date , progress=False)

    def calculate_rsi(data, period=14):
        def rma(series, period):
            alpha = 1 / period
            rma_values = [series[0]]
            for price in series[1:]:
                rma_values.append(alpha * price + (1 - alpha) * rma_values[-1])
            return np.array(rma_values)
        delta = data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = rma(gain, period)
        avg_loss = rma(loss, period)
        rs = avg_gain / avg_loss
        rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
        return rsi

    df['RSI'] = calculate_rsi(df,period=14)

    def calculate_atr(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    df['ATR'] = calculate_atr(df, period=atr_period)

    def rw_top(data, curr_index, order):
        if curr_index < order * 2 + 1:
            return False
        top = True
        k = curr_index - order
        v = data[k]
        for i in range(1, order + 1):
            if data[k + i] > v or data[k - i] > v:
                top = False
                break
        return top

    def rw_bottom(data, curr_index, order):
        if curr_index < order * 2 + 1:
            return False
        bottom = True
        k = curr_index - order
        v = data[k]
        for i in range(1, order + 1):
            if data[k + i] < v or data[k - i] < v:
                bottom = False
                break
        return bottom

    def rw_extremes(data, order):
        tops = []
        bottoms = []
        for i in range(len(data)):
            if rw_top(data, i, order):
                tops.append([i, i - order, data[i - order]])
            if rw_bottom(data, i, order):
                bottoms.append([i, i - order, data[i - order]])
        return tops, bottoms

    tops, bottoms = rw_extremes(df['Close'].to_numpy(), order)

    def format_tops_and_bottoms_separately(data, tops, bottoms):
        Top_records = []
        for idx, top in enumerate(tops, start=1):
            date = data.index[top[1]].strftime('%Y-%m-%d')
            price = round(float(top[2]), 3)
            rsi_value = round(data['RSI'].iloc[top[1]], 2)
            Top_records.append({'bottom/Top': f'Top {idx}', 'Date': date, 'Price': price, 'RSI': rsi_value})
        Top_df = pd.DataFrame(Top_records)

        bottom_records = []
        for idx, bottom in enumerate(bottoms, start=1):
            date = data.index[bottom[1]].strftime('%Y-%m-%d')
            price = round(float(bottom[2]), 3)
            rsi_value = round(data['RSI'].iloc[bottom[1]], 2)
            bottom_records.append({'bottom/Top': f'bottom {idx}', 'Date': date, 'Price': price, 'RSI': rsi_value})
        bottom_df = pd.DataFrame(bottom_records)

        combined_df = pd.concat([Top_df, bottom_df], axis=1)
        combined_df.columns = ['bottom/Top (Top)', 'Date (Top)', 'Price (Top)', 'RSI (Top)',
                               'bottom/Top (bottom)', 'Date (bottom)', 'Price (bottom)', 'RSI (bottom)']
        return combined_df

    formatted_df = format_tops_and_bottoms_separately(df, tops, bottoms)

    def positive_div(tops_and_bottoms):
        positive_div_list = []
        for i in range(len(tops_and_bottoms) - 1):
            bottom1 = tops_and_bottoms.iloc[i]
            bottom2 = tops_and_bottoms.iloc[i + 1]
            if bottom1['Price (bottom)'] > bottom2['Price (bottom)']:
                if bottom1['RSI (bottom)'] < bottom2['RSI (bottom)']:
                    positive_div_list.append({
                        'Date 1': bottom1['Date (bottom)'],
                        'Price 1': bottom1['Price (bottom)'],
                        'RSI 1': bottom1['RSI (bottom)'],
                        'Date 2': bottom2['Date (bottom)'],
                        'Price 2': bottom2['Price (bottom)'],
                        'RSI 2': bottom2['RSI (bottom)'],
                        'Type': 'Positive Divergence'
                    })
        return pd.DataFrame(positive_div_list)

    def negative_div(tops_and_bottoms):
        negative_div_list = []
        for i in range(len(tops_and_bottoms) - 1):
            top1 = tops_and_bottoms.iloc[i]
            top2 = tops_and_bottoms.iloc[i + 1]
            if top2['Price (Top)'] > top1['Price (Top)']:
                if top2['RSI (Top)'] < top1['RSI (Top)']:
                    negative_div_list.append({
                        'Date 1': top1['Date (Top)'],
                        'Price 1': top1['Price (Top)'],
                        'RSI 1': top1['RSI (Top)'],
                        'Date 2': top2['Date (Top)'],
                        'Price 2': top2['Price (Top)'],
                        'RSI 2': top2['RSI (Top)'],
                        'Type': 'Negative Divergence'
                    })
        return pd.DataFrame(negative_div_list)

    positive_div_df = positive_div(formatted_df)
    negative_div_df = negative_div(formatted_df)

    combined_df = pd.concat([positive_div_df, negative_div_df], ignore_index=True)
    combined_df.sort_values(by='Date 1', inplace=True, ignore_index=True)
    combined_df['Target Price'] = combined_df.apply(
        lambda row: row['Price 2'] + (1.8) * df.loc[row['Date 2'], 'ATR']
        if row['Type'] == 'Positive Divergence' else row['Price 2'] - (1.8) * df.loc[row['Date 2'], 'ATR'], axis=1
    )
    combined_df['Stop Loss'] = combined_df.apply(
        lambda row: row['Price 2'] - 1.35 * df.loc[row['Date 2'], 'ATR']
        if row['Type'] == 'Positive Divergence' else row['Price 2'] + 1.35 * df.loc[row['Date 2'], 'ATR'], axis=1
    )

    combined_df['ATR'] = combined_df['Date 2'].apply(
        lambda date: df.loc[date, 'ATR'] if date in df.index else np.nan
    )
    return combined_df

if __name__ == "__main__":
    ticker = 'TCELL.IS'
    combined_df = rsi_div(ticker)
    print(ticker)
    print(combined_df.tail())
    