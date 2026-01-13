import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="Calculadora KS-PME", layout="wide")

st.title("📊 Calculadora KS-PME (Kaplan-Schoar)")
st.markdown("""
Esta ferramenta calcula o **Public Market Equivalent** comparando os fluxos de um fundo de Private Equity 
com um índice de mercado público (Benchmark).

**Atualização:** - Ajuste automático de amortizações.
- Gráfico Multi-Linha: Cota vs TVPI vs Benchmark.
""")

# --- FUNÇÕES AUXILIARES ---

def clean_numeric_series(series):
    """
    Limpa uma coluna para garantir que seja numérica float.
    Trata formatação brasileira (1.000,00) e símbolos monetários.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    s = series.astype(str)
    s = s.str.replace('R$', '', regex=False)
    s = s.str.replace('$', '', regex=False)
    s = s.str.strip()
    
    def converter_numero(x):
        if ',' in x:
            x = x.replace('.', '').replace(',', '.')
        return x

    s = s.apply(converter_numero)
    return pd.to_numeric(s, errors='coerce').fillna(0)

def ajustar_amortizacao_pela_cota(df, col_map, threshold_percent=0.02):
    """
    Identifica amortizações implícitas na queda da cota e preenche a coluna Resgate.
    """
    col_cota = col_map['quota']
    col_pl = col_map['nav']
    col_resgate = col_map['dist']
    
    # 1. Calcular Quantidade de Cotas Implícita
    qtd_cotas = df[col_pl] / df[col_cota]
    qtd_cotas = qtd_cotas.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 2. Variação da cota
    cota_shift = df[col_cota].shift(1)
    var_cota_pct = (df[col_cota] / cota_shift) - 1
    var_cota_pct = var_cota_pct.fillna(0)
    
    # 3. Identificar Amortização (Queda > threshold)
    is_amortizacao = (var_cota_pct < -threshold_percent)
    
    # 4. Calcular valor financeiro
    diff_cota = cota_shift - df[col_cota]
    amortizacao_calculada = diff_cota * qtd_cotas
    amortizacao_calculada = amortizacao_calculada.where(is_amortizacao, 0)
    
    # 5. Somar ao Resgate existente
    df[col_resgate] = df[col_resgate].fillna(0) + amortizacao_calculada.fillna(0)
    
    total_ajustado = amortizacao_calculada.sum()
    if total_ajustado > 0:
        st.success(f"⚠️ Ajuste Automático: R$ {total_ajustado:,.2f} identificados como amortização e somados aos resgates.")
        
    return df

@st.cache_data
def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        
        date_cols = [c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()]
        if not date_cols:
            st.error("Não encontrei coluna de Data.")
            return None, None
            
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(by=date_col)
        
        return df, date_col
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return None, None

def calculate_ks_pme(fund_df, bench_df, date_col_fund, date_col_bench, col_map):
    
    # Limpar bench price
    bench_price_col_orig = [c for c in bench_df.columns if c != date_col_bench][0]
    bench_df[bench_price_col_orig] = clean_numeric_series(bench_df[bench_price_col_orig])

    # 1. Merge
    merged = pd.merge_asof(
        fund_df.sort_values(date_col_fund),
        bench_df.sort_values(date_col_bench),
        left_on=date_col_fund,
        right_on=date_col_bench,
        direction='backward'
    )
    merged = merged.rename(columns={bench_price_col_orig: 'bench_price'}).dropna(subset=['bench_price'])

    # 2. PME Math
    idx_T = merged['bench_price'].iloc[-1]
    merged['idx_multiplier'] = idx_T / merged['bench_price']
    
    merged['fv_call'] = merged[col_map['call']] * merged['idx_multiplier']
    merged['fv_dist'] = merged[col_map['dist']] * merged['idx_multiplier']
    
    nav_final = merged[col_map['nav']].iloc[-1]
    
    numerator = merged['fv_dist'].sum() + nav_final
    denominator = merged['fv_call'].sum()
    
    ks_pme = numerator / denominator if denominator != 0 else 0
    
    # --- CÁLCULO DE MÉTRICAS ACUMULADAS (SÉRIE TEMPORAL) ---
    
    # 1. Acumulados
    merged['cum_dist'] = merged[col_map['dist']].cumsum()
    merged['cum_call'] = merged[col_map['call']].cumsum()
    
    # 2. Capital Investido (Dinâmico)
    # Se cum_call for 0 (sem chamadas registradas), usa NAV inicial como base de custo
    start_nav = merged[col_map['nav']].iloc[0]
    merged['invested_capital'] = merged['cum_call'].replace(0, np.nan).fillna(start_nav)
    merged['invested_capital'] = merged['invested_capital'].replace(0, 1) # Proteção final div/0
    
    # 3. TVPI Série Temporal (Base 100 para o gráfico)
    # TVPI = (Distribuído + NAV Atual) / Capital Investido
    merged['TVPI_Series'] = (merged['cum_dist'] + merged[col_map['nav']]) / merged['invested_capital']
    merged['TVPI_Base100'] = merged['TVPI_Series'] * 100
    
    # --- RETORNOS FINAIS ---
    
    total_dist = merged[col_map['dist']].sum()
    total_call = merged[col_map['call']].sum()
    invested_capital_final = total_call if total_call > 0 else start_nav
    
    if invested_capital_final > 0:
        fund_return = ((total_dist + nav_final) / invested_capital_final) - 1
    else:
        fund_return = 0

    start_bench = merged['bench_price'].iloc[0]
    end_bench = merged['bench_price'].iloc[-1]
    bench_return = (end_bench / start_bench) - 1 if start_bench != 0 else 0
    
    return ks_pme, fund_return, bench_return, merged

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
        st.subheader("Mapeamento de Colunas")
        
        c1, c2, c3, c4 = st.columns(4)
        cols = df_fund.columns.tolist()
        
        def get_index(options, keywords):
            for i, opt in enumerate(options):
                if any(k in opt.lower() for k in keywords):
                    return i
            return 0

        col_quota = c1.selectbox("Cota", cols, index=get_index(cols, ['cota', 'quota']))
        col_nav = c2.selectbox("PL (NAV)", cols, index=get_index(cols, ['pl', 'nav', 'patrimonio']))
        col_call = c3.selectbox("Captação (Call)", cols, index=get_index(cols, ['capta', 'call']))
        col_dist = c4.selectbox("Resgates (Dist)", cols, index=get_index(cols, ['resga', 'dist']))
        
        col_map = {'quota': col_quota, 'nav': col_nav, 'call': col_call, 'dist': col_dist}
        
        if st.button("Calcular KS-PME", type="primary"):
            try:
                # 1. Limpeza
                for col in [col_quota, col_nav, col_call, col_dist]:
                    df_fund[col] = clean_numeric_series(df_fund[col])

                # 2. Ajuste Amortização
                df_fund = ajustar_amortizacao_pela_cota(df_fund, col_map, threshold_percent=0.02)

                # 3. Cálculo
                ks_pme, fund_ret, bench_ret, df_proc = calculate_ks_pme(
                    df_fund, df_bench, date_col_fund, date_col_bench, col_map
                )
                
                st.divider()
                
                # Métricas
                m1, m2, m3 = st.columns(3)
                m1.metric("KS-PME Score", f"{ks_pme:.2f}x", delta="Alpha Positivo" if ks_pme > 1 else "Alpha Negativo")
                m2.metric("Retorno Total Fundo (TVPI)", f"{fund_ret:.2%}", help="Múltiplo final de todo capital investido")
                m3.metric("Retorno Benchmark", f"{bench_ret:.2%}")
                
                # Gráfico
                st.subheader("Análise de Performance (Base 100)")
                
                # Normalizações Base 100
                df_proc['Fundo_Norm'] = (df_proc[col_map['quota']] / df_proc[col_map['quota']].iloc[0]) * 100
                df_proc['Bench_Norm'] = (df_proc['bench_price'] / df_proc['bench_price'].iloc[0]) * 100
                # TVPI_Base100 já foi calculado na função
                
                fig = go.Figure()
                
                # Linha 1: Cota (Azul) - Mostra a desvalorização nominal
                fig.add_trace(go.Scatter(
                    x=df_proc[date_col_fund], 
                    y=df_proc['Fundo_Norm'], 
                    name='Valor Cota',
                    line=dict(color='blue', width=1, dash='dot')
                ))
                
                # Linha 2: TVPI (Verde) - Mostra a rentabilidade real (distribuição + nav)
                fig.add_trace(go.Scatter(
                    x=df_proc[date_col_fund], 
                    y=df_proc['TVPI_Base100'], 
                    name='Retorno Total (TVPI)',
                    line=dict(color='green', width=3)
                ))
                
                # Linha 3: Benchmark (Laranja/Vermelho)
                fig.add_trace(go.Scatter(
                    x=df_proc[date_col_fund], 
                    y=df_proc['Bench_Norm'], 
                    name='Benchmark',
                    line=dict(color='firebrick', width=2)
                ))
                
                fig.update_layout(
                    title="Cota Nominal vs. Retorno Total vs. Benchmark",
                    hovermode="x unified",
                    yaxis_title="Base 100 (Início = 100)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Dica de Leitura:** A linha tracejada (Cota) cai quando o fundo devolve dinheiro. A linha verde (TVPI) soma esse dinheiro devolvido ao valor restante, mostrando a verdadeira criação de valor.")

                with st.expander("Ver dados processados"):
                    st.dataframe(df_proc)
                    
            except Exception as e:
                st.error(f"Erro: {e}")
