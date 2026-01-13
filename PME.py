import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="Calculadora KS-PME", layout="wide")

st.title("📊 Calculadora KS-PME (Kaplan-Schoar)")
st.markdown("""
Esta ferramenta compara a performance de um fundo de Private Equity com um Benchmark de mercado público.

**Conceito (Apples to Apples):**
Comparamos o retorno do seu fundo com uma **carteira teórica** que compra o índice (Benchmark)
exatamente nas mesmas datas e proporções das chamadas de capital do fundo.
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
    
    # 1. Quantidade de Cotas
    qtd_cotas = df[col_pl] / df[col_cota]
    qtd_cotas = qtd_cotas.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 2. Variação
    cota_shift = df[col_cota].shift(1)
    var_cota_pct = (df[col_cota] / cota_shift) - 1
    
    # 3. Detectar Amortização
    is_amortizacao = (var_cota_pct.fillna(0) < -threshold_percent)
    
    # 4. Calcular Financeiro
    diff_cota = cota_shift - df[col_cota]
    amortizacao_calculada = diff_cota * qtd_cotas
    amortizacao_calculada = amortizacao_calculada.where(is_amortizacao, 0)
    
    # 5. Aplicar
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

def calculate_ks_pme(fund_df, bench_df, date_col_fund, date_col_bench, col_map):
    # Limpar Benchmark Price
    bench_price_col = [c for c in bench_df.columns if c != date_col_bench][0]
    bench_df[bench_price_col] = clean_numeric_series(bench_df[bench_price_col])

    # 1. Merge (Datas do Fundo mandam)
    merged = pd.merge_asof(
        fund_df.sort_values(date_col_fund),
        bench_df.sort_values(date_col_bench),
        left_on=date_col_fund,
        right_on=date_col_bench,
        direction='backward'
    ).rename(columns={bench_price_col: 'bench_price'}).dropna(subset=['bench_price'])

    # 2. PME Tradicional (Kaplan-Schoar) - Math
    idx_T = merged['bench_price'].iloc[-1]
    merged['idx_multiplier'] = idx_T / merged['bench_price']
    
    merged['fv_call'] = merged[col_map['call']] * merged['idx_multiplier']
    merged['fv_dist'] = merged[col_map['dist']] * merged['idx_multiplier']
    
    nav_final = merged[col_map['nav']].iloc[-1]
    numerator = merged['fv_dist'].sum() + nav_final
    denominator = merged['fv_call'].sum()
    
    ks_pme = numerator / denominator if denominator != 0 else 0
    
    # 3. Séries Temporais para Gráfico e Métricas PME
    
    # --- A. Fundo Real ---
    merged['cum_dist'] = merged[col_map['dist']].cumsum()
    merged['cum_call'] = merged[col_map['call']].cumsum()
    
    # Base de Custo Dinâmica
    start_nav = merged[col_map['nav']].iloc[0]
    merged['invested_capital'] = merged['cum_call'].replace(0, np.nan).fillna(start_nav)
    
    # TVPI Fundo = (Distribuições + NAV Atual) / Capital Investido
    merged['TVPI_Fund'] = (merged['cum_dist'] + merged[col_map['nav']]) / merged['invested_capital']
    
    # --- B. Benchmark Equivalente (PME Simulation) ---
    # Simulamos comprar o índice com os Calls e vender com os Dists
    
    # Fluxo de Cotas do Índice: (Compra com Call - Vende com Dist) / Preço do Índice
    merged['flow_shares'] = (merged[col_map['call']] - merged[col_map['dist']]) / merged['bench_price']
    
    # Se não houve Calls, assume que o NAV inicial estava comprado no índice
    total_call = merged[col_map['call']].sum()
    initial_shares = (start_nav / merged['bench_price'].iloc[0]) if total_call == 0 else 0
    
    merged['cum_shares_bench'] = merged['flow_shares'].cumsum() + initial_shares
    
    # PME NAV = Quantidade de Cotas Teóricas * Preço Atual
    merged['PME_NAV'] = merged['cum_shares_bench'] * merged['bench_price']
    
    # TVPI Bench = (Distribuições Mesmas do Fundo + PME NAV) / Capital Investido
    merged['TVPI_Bench'] = (merged['cum_dist'] + merged['PME_NAV']) / merged['invested_capital']
    
    # Normalização Base 100 para o gráfico
    merged['TVPI_Fund_100'] = merged['TVPI_Fund'] * 100
    merged['TVPI_Bench_100'] = merged['TVPI_Bench'] * 100
    merged['Quota_100'] = (merged[col_map['quota']] / merged[col_map['quota']].iloc[0]) * 100
    
    # Retornos Finais
    fund_ret_final = merged['TVPI_Fund'].iloc[-1] - 1
    bench_ret_pme = merged['TVPI_Bench'].iloc[-1] - 1

    return ks_pme, fund_ret_final, bench_ret_pme, merged

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
        st.subheader("Mapeamento")
        
        c1, c2, c3, c4 = st.columns(4)
        cols = df_fund.columns.tolist()
        
        # Helpers de busca
        def get_idx(opts, keys):
            for i, o in enumerate(opts):
                if any(k in o.lower() for k in keys): return i
            return 0

        col_q = c1.selectbox("Cota", cols, index=get_idx(cols, ['cota', 'quota']))
        col_n = c2.selectbox("PL (NAV)", cols, index=get_idx(cols, ['pl', 'nav', 'patrimonio']))
        col_c = c3.selectbox("Captação (Call)", cols, index=get_idx(cols, ['capta', 'call']))
        col_d = c4.selectbox("Resgates (Dist)", cols, index=get_idx(cols, ['resga', 'dist']))
        
        col_map = {'quota': col_q, 'nav': col_n, 'call': col_c, 'dist': col_d}
        
        if st.button("Calcular KS-PME", type="primary"):
            try:
                # 1. Limpeza
                for c in col_map.values(): df_fund[c] = clean_numeric_series(df_fund[c])
                
                # 2. Ajuste
                df_fund = ajustar_amortizacao_pela_cota(df_fund, col_map, 0.02)
                
                # 3. Cálculo
                ks, ret_f, ret_b_pme, df_res = calculate_ks_pme(df_fund, df_bench, date_col_fund, date_col_bench, col_map)
                
                st.divider()
                
                # KPIs
                k1, k2, k3 = st.columns(3)
                k1.metric("KS-PME Score", f"{ks:.2f}x", delta="Alpha Gerado" if ks > 1 else "Alpha Negativo")
                k2.metric("Retorno Fundo (TVPI)", f"{ret_f:.2%}", help="Retorno total real do fundo")
                k3.metric("Retorno Benchmark (PME)", f"{ret_b_pme:.2%}", help="Retorno da carteira equivalente no índice")
                
                # --- VEREDITO PEDAGÓGICO ---
                if ks > 1.0:
                    st.success(f"🚀 **O Fundo superou o Benchmark!**\n\nO índice de **{ks:.2f}x** significa que o fundo entregou **{(ks-1)*100:.1f}% mais riqueza** do que se o mesmo capital tivesse sido investido no índice público (PME).")
                elif ks < 1.0:
                    st.error(f"📉 **O Fundo perdeu para o Benchmark.**\n\nO índice de **{ks:.2f}x** significa que o fundo entregou apenas **{ks*100:.1f}%** da riqueza que o investidor teria acumulado no índice público.")
                else:
                    st.warning(f"⚖️ **Empate Técnico.**\n\nO fundo entregou exatamente o mesmo retorno do índice de referência.")

                # Gráfico
                st.subheader("Performance Relativa (TVPI Base 100)")
                
                fig = go.Figure()
                
                # Fundo TVPI (Verde)
                fig.add_trace(go.Scatter(
                    x=df_res[date_col_fund], y=df_res['TVPI_Fund_100'],
                    name='Fundo (TVPI Real)', line=dict(color='#2ca02c', width=3)
                ))
                
                # Benchmark PME TVPI (Laranja)
                fig.add_trace(go.Scatter(
                    x=df_res[date_col_fund], y=df_res['TVPI_Bench_100'],
                    name='Benchmark PME (Equivalente)', line=dict(color='#ff7f0e', width=3)
                ))
                
                # Cota Nominal (Azul Tracejado)
                fig.add_trace(go.Scatter(
                    x=df_res[date_col_fund], y=df_res['Quota_100'],
                    name='Valor da Cota (Ref)', line=dict(color='blue', width=1, dash='dot')
                ))

                fig.update_layout(
                    title="Criação de Valor: Fundo vs Benchmark (Fluxos Equivalentes)",
                    yaxis_title="Base 100",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Ver dados processados"):
                    st.dataframe(df_res)
                    
            except Exception as e:
                st.error(f"Erro fatal: {e}")
