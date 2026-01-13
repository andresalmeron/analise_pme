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

**Atualização:** Inclui ajuste automático para amortizações via redução de cota (FIPs).
""")

# --- FUNÇÕES AUXILIARES ---

def clean_numeric_series(series):
    """
    Limpa uma coluna para garantir que seja numérica float.
    Trata formatação brasileira (1.000,00) e símbolos monetários.
    """
    # Se já for numérico, retorna direto
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # Converte para string para poder manipular
    s = series.astype(str)
    
    # Remove R$, $, espaços e pontos de milhar (assumindo formato BR: 1.000,00 se houver vírgula)
    s = s.str.replace('R$', '', regex=False)
    s = s.str.replace('$', '', regex=False)
    s = s.str.strip()
    
    # Lógica para detectar se é formato BR (tem vírgula como decimal)
    # Se tiver vírgula, assumimos que é decimal -> trocamos por ponto
    # E removemos os pontos existentes (milhares) antes
    def converter_numero(x):
        if ',' in x:
            # Formato BR: 1.000,00 -> 1000.00
            x = x.replace('.', '').replace(',', '.')
        return x

    s = s.apply(converter_numero)
    
    return pd.to_numeric(s, errors='coerce').fillna(0)

def ajustar_amortizacao_pela_cota(df, col_map, threshold_percent=0.02):
    """
    Identifica amortizações implícitas na queda da cota e preenche a coluna Resgate.
    """
    # Mapear colunas baseadas na seleção do usuário
    col_cota = col_map['quota']
    col_pl = col_map['nav']
    col_resgate = col_map['dist']
    
    # 1. Calcular Quantidade de Cotas Implícita (Patrimônio / Cota)
    # Usamos replace de inf caso divida por zero
    qtd_cotas = df[col_pl] / df[col_cota]
    qtd_cotas = qtd_cotas.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 2. Calcular variação da cota em relação ao dia anterior
    cota_shift = df[col_cota].shift(1) # Valor do dia anterior
    
    # Evitar divisão por zero na variação
    var_cota_pct = (df[col_cota] / cota_shift) - 1
    var_cota_pct = var_cota_pct.fillna(0)
    
    # 3. Identificar Amortização (Queda brusca na cota maior que o threshold)
    is_amortizacao = (var_cota_pct < -threshold_percent)
    
    # 4. Calcular o valor financeiro da amortização
    # Valor = (Queda na Cota) * (Quantidade de Cotas do dia)
    diff_cota = cota_shift - df[col_cota]
    amortizacao_calculada = diff_cota * qtd_cotas
    
    # Filtrar apenas onde é amortização positiva
    amortizacao_calculada = amortizacao_calculada.where(is_amortizacao, 0)
    
    # 5. Aplicar o ajuste na coluna Resgate
    # Se a coluna resgate tiver nulos ou zeros, somamos o valor calculado
    df[col_resgate] = df[col_resgate].fillna(0) + amortizacao_calculada.fillna(0)
    
    # Feedback visual se houve ajuste
    total_ajustado = amortizacao_calculada.sum()
    if total_ajustado > 0:
        st.success(f"⚠️ Ajuste Automático: Detectamos {is_amortizacao.sum()} amortizações via cota. Total adicionado aos resgates: R$ {total_ajustado:,.2f}")
        
    return df

@st.cache_data
def load_csv_data(uploaded_file):
    """Carrega dados de CSV para DataFrame."""
    try:
        # Lê o CSV
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # Limpeza nomes das colunas
        df.columns = df.columns.str.strip()
        
        # Identificar coluna de data
        date_cols = [c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()]
        
        if not date_cols:
            st.error("Não encontrei uma coluna com nome 'data' ou 'date'. Verifique o cabeçalho.")
            return None, None
            
        date_col = date_cols[0]
        
        # Converte para datetime
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(by=date_col)
        
        return df, date_col
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        return None, None

def calculate_ks_pme(fund_df, bench_df, date_col_fund, date_col_bench, col_map):
    """Realiza o cálculo do KS-PME."""
    
    # Nota: A limpeza numérica já foi feita antes desta função ser chamada
    
    # Limpar coluna de preço do benchmark
    bench_price_col_orig = [c for c in bench_df.columns if c != date_col_bench][0]
    bench_df[bench_price_col_orig] = clean_numeric_series(bench_df[bench_price_col_orig])

    # 1. Merge dos dados
    merged = pd.merge_asof(
        fund_df.sort_values(date_col_fund),
        bench_df.sort_values(date_col_bench),
        left_on=date_col_fund,
        right_on=date_col_bench,
        direction='backward'
    )
    
    # Renomear bench price
    merged = merged.rename(columns={bench_price_col_orig: 'bench_price'})
    
    if merged['bench_price'].isnull().any():
        st.warning("Atenção: Existem datas sem cotação correspondente no Benchmark.")
        merged = merged.dropna(subset=['bench_price'])

    # 2. Definir o valor final do índice (T)
    idx_T = merged['bench_price'].iloc[-1]
    
    # 3. Fatores de Ajuste
    merged['idx_multiplier'] = idx_T / merged['bench_price']
    
    # 4. Valores Futuros (FV)
    merged['fv_call'] = merged[col_map['call']] * merged['idx_multiplier']
    merged['fv_dist'] = merged[col_map['dist']] * merged['idx_multiplier']
    
    # 5. Numerador e Denominador
    nav_final = merged[col_map['nav']].iloc[-1]
    
    numerator = merged['fv_dist'].sum() + nav_final # O multiplicador do NAV final é 1.0
    denominator = merged['fv_call'].sum()
    
    ks_pme = numerator / denominator if denominator != 0 else 0
    
    # --- RETORNOS ---
    start_quota = merged[col_map['quota']].iloc[0]
    end_quota = merged[col_map['quota']].iloc[-1]
    
    # Proteção contra divisão por zero na cota
    fund_return = (end_quota / start_quota) - 1 if start_quota != 0 else 0
    
    start_bench = merged['bench_price'].iloc[0]
    end_bench = merged['bench_price'].iloc[-1]
    bench_return = (end_bench / start_bench) - 1 if start_bench != 0 else 0
    
    return ks_pme, fund_return, bench_return, merged

# --- INTERFACE ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Fundo (CSV)")
    fund_file = st.file_uploader("Dados do Fundo", type=['csv'])

with col2:
    st.subheader("2. Upload Benchmark (CSV)")
    bench_file = st.file_uploader("Dados do Benchmark", type=['csv'])

if fund_file and bench_file:
    df_fund, date_col_fund = load_csv_data(fund_file)
    df_bench, date_col_bench = load_csv_data(bench_file)
    
    if df_fund is not None and df_bench is not None:
        
        st.divider()
        st.subheader("Mapeamento de Colunas")
        
        c1, c2, c3, c4 = st.columns(4)
        
        def get_index(options, keywords):
            for i, opt in enumerate(options):
                if any(k in opt.lower() for k in keywords):
                    return i
            return 0

        cols = df_fund.columns.tolist()
        
        col_quota = c1.selectbox("Cota", cols, index=get_index(cols, ['cota', 'quota']))
        col_nav = c2.selectbox("PL (NAV)", cols, index=get_index(cols, ['pl', 'nav', 'patrimonio']))
        col_call = c3.selectbox("Captação (Call)", cols, index=get_index(cols, ['capta', 'call', 'aport']))
        col_dist = c4.selectbox("Resgates (Dist)", cols, index=get_index(cols, ['resga', 'dist', 'pagamento']))
        
        col_map = {'quota': col_quota, 'nav': col_nav, 'call': col_call, 'dist': col_dist}
        
        if st.button("Calcular KS-PME", type="primary"):
            try:
                # 1. Limpeza forçada dos dados antes de qualquer lógica
                cols_to_clean = [col_quota, col_nav, col_call, col_dist]
                for col in cols_to_clean:
                    df_fund[col] = clean_numeric_series(df_fund[col])

                # 2. Aplicar Ajuste de Amortização (Correção do -98%)
                # Threshold de 2% (0.02) captura quedas relevantes que são amortizações
                df_fund = ajustar_amortizacao_pela_cota(df_fund, col_map, threshold_percent=0.02)

                # 3. Calcular KS-PME
                ks_pme, fund_ret, bench_ret, df_processed = calculate_ks_pme(
                    df_fund, df_bench, date_col_fund, date_col_bench, col_map
                )
                
                st.divider()
                
                # Métricas
                m1, m2, m3 = st.columns(3)
                m1.metric("KS-PME Score", f"{ks_pme:.2f}x", delta="Alpha Positivo" if ks_pme > 1 else "Alpha Negativo")
                m2.metric("Retorno Fundo (Cota)", f"{fund_ret:.2%}")
                m3.metric("Retorno Benchmark", f"{bench_ret:.2%}")
                
                st.markdown(f"**Resultado:** Para cada R$ 1,00 investido, o fundo retornou o equivalente a **R$ {ks_pme:.2f}** no benchmark ajustado.")
                
                # Gráfico
                st.subheader("Curva de Performance (Base 100)")
                # Normalização base 100
                df_processed['Fundo_Norm'] = (df_processed[col_map['quota']] / df_processed[col_map['quota']].iloc[0]) * 100
                df_processed['Bench_Norm'] = (df_processed['bench_price'] / df_processed['bench_price'].iloc[0]) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_processed[date_col_fund], y=df_processed['Fundo_Norm'], name='Fundo (Cota)'))
                fig.add_trace(go.Scatter(x=df_processed[date_col_fund], y=df_processed['Bench_Norm'], name='Benchmark'))
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Ver dados processados (inclui amortizações ajustadas)"):
                    st.dataframe(df_processed)
                    
            except Exception as e:
                st.error(f"Erro durante o cálculo: {e}")
                st.write("Dica: Verifique se as colunas numéricas contêm apenas números.")
