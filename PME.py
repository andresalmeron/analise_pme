import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="Calculadora KS-PME | Portfel", layout="wide")

st.title("📊 Calculadora KS-PME (Kaplan-Schoar)")
st.markdown("""
Esta ferramenta compara a performance de um fundo de Private Equity com um Benchmark de mercado público.
O cálculo utiliza fluxos de caixa reais (Calls e Dists) para criar uma comparação justa.
""")

# --- FUNÇÕES AUXILIARES ---

def clean_numeric_series(series):
    """Limpeza robusta para garantir números floats (suporta BR e US)."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    s = series.astype(str)
    s = s.str.replace('R$', '', regex=False).str.replace('$', '', regex=False).str.strip()
    
    def converter_numero(x):
        if ',' in x: # Formato BR
            x = x.replace('.', '').replace(',', '.')
        return x

    s = s.apply(converter_numero)
    return pd.to_numeric(s, errors='coerce').fillna(0)

def ajustar_amortizacao_pela_cota(df, col_map, threshold_percent=0.02):
    """Identifica amortizações implícitas na queda da cota."""
    col_cota, col_pl, col_resgate = col_map['quota'], col_map['nav'], col_map['dist']
    
    qtd_cotas = df[col_pl] / df[col_cota]
    qtd_cotas = qtd_cotas.replace([np.inf, -np.inf], 0).fillna(0)
    
    cota_shift = df[col_cota].shift(1)
    var_cota_pct = (df[col_cota] / cota_shift) - 1
    is_amortizacao = (var_cota_pct.fillna(0) < -threshold_percent)
    
    diff_cota = cota_shift - df[col_cota]
    amortizacao_calculada = diff_cota * qtd_cotas
    amortizacao_calculada = amortizacao_calculada.where(is_amortizacao, 0)
    
    df[col_resgate] = df[col_resgate].fillna(0) + amortizacao_calculada.fillna(0)
    
    total_ajustado = amortizacao_calculada.sum()
    if total_ajustado > 0:
        st.success(f"⚠️ Ajuste Automático: R$ {total_ajustado:,.2f} detectados como amortização (queda de cota).")
        
    return df

@st.cache_data
def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        date_cols = [c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()]
        
        if not date_cols:
            st.error("Coluna de Data não encontrada.")
            return None, None
            
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        return df.dropna(subset=[date_col]).sort_values(by=date_col), date_col
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return None, None

def calculate_ks_pme(fund_df, bench_df, date_col_fund, date_col_bench, col_map, bench_fee_annual=0.002):
    bench_price_col = [c for c in bench_df.columns if c != date_col_bench][0]
    bench_df[bench_price_col] = clean_numeric_series(bench_df[bench_price_col])

    merged = pd.merge_asof(
        fund_df.sort_values(date_col_fund),
        bench_df.sort_values(date_col_bench),
        left_on=date_col_fund,
        right_on=date_col_bench,
        direction='backward'
    ).rename(columns={bench_price_col: 'bench_price_orig'}).dropna(subset=['bench_price_orig'])

    # Ajuste de custo ETF
    start_date = merged[date_col_fund].iloc[0]
    merged['days_since_start'] = (merged[date_col_fund] - start_date).dt.days
    merged['bench_price'] = merged['bench_price_orig'] * ((1 - bench_fee_annual) ** (merged['days_since_start'] / 365.25))

    idx_T = merged['bench_price'].iloc[-1]
    merged['idx_multiplier'] = idx_T / merged['bench_price']
    merged['fv_call'] = merged[col_map['call']] * merged['idx_multiplier']
    merged['fv_dist'] = merged[col_map['dist']] * merged['idx_multiplier']
    
    nav_final = merged[col_map['nav']].iloc[-1]
    numerator = merged['fv_dist'].sum() + nav_final
    denominator = merged['fv_call'].sum()
    ks_pme = numerator / denominator if denominator != 0 else 0
    
    # Séries Temporais
    merged['cum_dist'] = merged[col_map['dist']].cumsum()
    merged['cum_call'] = merged[col_map['call']].cumsum()
    start_nav = merged[col_map['nav']].iloc[0]
    merged['invested_capital'] = merged['cum_call'].replace(0, np.nan).fillna(start_nav)
    
    merged['TVPI_Fund'] = (merged['cum_dist'] + merged[col_map['nav']]) / merged['invested_capital']
    merged['flow_shares'] = (merged[col_map['call']] - merged[col_map['dist']]) / merged['bench_price']
    initial_shares = (start_nav / merged['bench_price'].iloc[0]) if merged[col_map['call']].sum() == 0 else 0
    merged['cum_shares_bench'] = merged['flow_shares'].cumsum() + initial_shares
    merged['PME_NAV'] = merged['cum_shares_bench'] * merged['bench_price']
    merged['TVPI_Bench'] = (merged['cum_dist'] + merged['PME_NAV']) / merged['invested_capital']
    
    merged['Years_Elapsed'] = (merged[date_col_fund] - start_date).dt.days / 365.25
    def calc_cagr(tvpi, years):
        if years < 0.1: return 0
        return (max(0, tvpi) ** (1/years)) - 1
        
    merged['CAGR_Fund'] = merged.apply(lambda x: calc_cagr(x['TVPI_Fund'], x['Years_Elapsed']), axis=1) * 100
    merged['CAGR_Bench'] = merged.apply(lambda x: calc_cagr(x['TVPI_Bench'], x['Years_Elapsed']), axis=1) * 100

    metrics = {
        'fund_tvpi': merged['TVPI_Fund'].iloc[-1],
        'bench_tvpi': merged['TVPI_Bench'].iloc[-1],
        'fund_100k': merged['TVPI_Fund'].iloc[-1] * 100000,
        'bench_100k': merged['TVPI_Bench'].iloc[-1] * 100000,
        'fund_cagr': merged['CAGR_Fund'].iloc[-1],
        'bench_cagr': merged['CAGR_Bench'].iloc[-1]
    }

    return ks_pme, metrics, merged

# --- INTERFACE ---

col1, col2 = st.columns(2)
with col1:
    fund_file = st.file_uploader("1. Dados do Fundo (CSV)", type=['csv'])
with col2:
    bench_file = st.file_uploader("2. Dados do Benchmark (CSV)", type=['csv'])

if fund_file and bench_file:
    df_fund, date_col_fund = load_csv_data(fund_file)
    df_bench, date_col_bench = load_csv_data(bench_file)
    
    if df_fund is not None and df_bench is not None:
        st.divider()
        st.subheader("Configuração e Mapeamento")
        
        f_c1, f_c2 = st.columns([1, 3])
        bench_fee_pct = f_c1.number_input("Taxa ETF (% a.a.)", value=0.20, step=0.05)

        c1, c2, c3, c4 = st.columns(4)
        cols = df_fund.columns.tolist()
        def get_idx(opts, keys):
            for i, o in enumerate(opts):
                if any(k in o.lower() for k in keys): return i
            return 0

        col_q = c1.selectbox("Cota", cols, index=get_idx(cols, ['cota']))
        col_n = c2.selectbox("PL (NAV)", cols, index=get_idx(cols, ['pl', 'nav', 'patrimonio']))
        col_c = c3.selectbox("Captação", cols, index=get_idx(cols, ['capta', 'call']))
        col_d = c4.selectbox("Resgates", cols, index=get_idx(cols, ['resga', 'dist']))
        
        if st.button("🚀 Calcular Performance Real", type="primary"):
            try:
                col_map = {'quota': col_q, 'nav': col_n, 'call': col_c, 'dist': col_d}
                for c in col_map.values(): df_fund[c] = clean_numeric_series(df_fund[c])
                df_fund = ajustar_amortizacao_pela_cota(df_fund, col_map, 0.02)
                
                ks, m, df_res = calculate_ks_pme(df_fund, df_bench, date_col_fund, date_col_bench, col_map, bench_fee_pct/100)
                
                # --- JANELA DE DESTAQUE ---
                st.divider()
                with st.container(border=True):
                    st.subheader("🎯 Veredito da Estratégia")
                    v1, v2 = st.columns([2, 3])
                    
                    with v1:
                        st.metric("KS-PME Score", f"{ks:.2f}x", 
                                  delta="Alpha Positivo" if ks > 1 else "Alpha Negativo",
                                  help="Valores acima de 1.0 indicam que o fundo superou o benchmark.")
                    
                    with v2:
                        if ks > 1.0:
                            st.success(f"""
                            **Performance Superior!** Para cada **R$ 1,00** investido no fundo, você obteve **R$ {m['fund_tvpi']:.2f}**.  
                            No benchmark, o retorno teria sido de apenas **R$ {m['bench_tvpi']:.2f}**.
                            """)
                        else:
                            st.error(f"""
                            **Performance Abaixo do Índice.** Para cada **R$ 1,00** investido no fundo, você obteve **R$ {m['fund_tvpi']:.2f}**.  
                            No benchmark, você teria acumulado **R$ {m['bench_tvpi']:.2f}**.
                            """)

                # Raio-X
                st.subheader("📊 Comparativo Detalhado")
                res_table = pd.DataFrame({
                    'Métrica': ['Múltiplo (TVPI)', 'Retorno Anual (CAGR)', 'Patrimônio Final (Base R$ 100k)'],
                    'Fundo PE': [f"{m['fund_tvpi']:.2f}x", f"{m['fund_cagr']:.1f}%", f"R$ {m['fund_100k']:,.2f}"],
                    'Benchmark PME': [f"{m['bench_tvpi']:.2f}x", f"{m['bench_cagr']:.1f}%", f"R$ {m['bench_100k']:,.2f}"]
                })
                st.table(res_table)
                
                # Gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_res[date_col_fund], y=df_res['TVPI_Fund']*100, name='Fundo (TVPI)', line=dict(color='#2ca02c', width=3)))
                fig.add_trace(go.Scatter(x=df_res[date_col_fund], y=df_res['TVPI_Bench']*100, name='Benchmark PME', line=dict(color='#ff7f0e', width=3)))
                fig.update_layout(title="Evolução da Riqueza (Base 100)", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro no processamento: {e}")
