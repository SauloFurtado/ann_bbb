import io
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Treinamento de MLP para Regressão",
    page_icon="🧠",
    layout="wide",
)


def converter_camadas(texto: str) -> tuple[int, ...]:
    """
    Converte:
        '100'       -> (100,)
        '100, 50'   -> (100, 50)
        '64,32,16'  -> (64, 32, 16)
    """
    partes = [parte.strip() for parte in texto.split(",") if parte.strip()]

    if not partes:
        raise ValueError("Informe pelo menos uma camada oculta.")

    camadas = tuple(int(parte) for parte in partes)

    if any(neuronios <= 0 for neuronios in camadas):
        raise ValueError(
            "A quantidade de neurônios deve ser maior que zero."
        )

    return camadas


def carregar_csv(
    arquivo,
    separador: str,
    decimal: str,
    codificacao: str,
) -> pd.DataFrame:
    arquivo.seek(0)

    if separador == "Detectar automaticamente":
        return pd.read_csv(
            arquivo,
            sep=None,
            engine="python",
            decimal=decimal,
            encoding=codificacao,
        )

    separadores = {
        "Vírgula": ",",
        "Ponto e vírgula": ";",
        "Tabulação": "\t",
        "Barra vertical": "|",
    }

    return pd.read_csv(
        arquivo,
        sep=separadores[separador],
        decimal=decimal,
        encoding=codificacao,
    )


st.title("🧠 Rede neural MLP para regressão")

st.write(
    """
    Envie um arquivo CSV, escolha a variável que será prevista e configure
    os hiperparâmetros do `MLPRegressor`.
    """
)

arquivo = st.file_uploader(
    "Selecione o arquivo CSV",
    type="csv",
    max_upload_size=50,
)

if arquivo is None:
    st.info("Envie um arquivo CSV para iniciar.")
    st.stop()


# -------------------------------------------------------------------------
# Configuração de leitura do CSV
# -------------------------------------------------------------------------

with st.expander("Configurações de leitura do CSV"):
    coluna_1, coluna_2, coluna_3 = st.columns(3)

    with coluna_1:
        separador = st.selectbox(
            "Separador das colunas",
            [
                "Detectar automaticamente",
                "Vírgula",
                "Ponto e vírgula",
                "Tabulação",
                "Barra vertical",
            ],
        )

    with coluna_2:
        decimal = st.selectbox(
            "Separador decimal",
            [".", ","],
        )

    with coluna_3:
        codificacao = st.selectbox(
            "Codificação",
            ["utf-8", "latin-1"],
        )


try:
    dados = carregar_csv(
        arquivo=arquivo,
        separador=separador,
        decimal=decimal,
        codificacao=codificacao,
    )
except Exception as erro:
    st.error(f"Não foi possível ler o arquivo CSV: {erro}")
    st.stop()


if dados.empty:
    st.error("O arquivo não contém registros.")
    st.stop()

if len(dados.columns) < 2:
    st.error(
        "O arquivo precisa ter pelo menos uma variável de entrada "
        "e uma variável-alvo."
    )
    st.stop()


# -------------------------------------------------------------------------
# Visualização dos dados
# -------------------------------------------------------------------------

st.subheader("1. Visualização dos dados")

metrica_1, metrica_2, metrica_3 = st.columns(3)

metrica_1.metric("Linhas", len(dados))
metrica_2.metric("Colunas", len(dados.columns))
metrica_3.metric(
    "Valores ausentes",
    int(dados.isna().sum().sum()),
)

st.dataframe(
    dados.head(20),
    use_container_width=True,
)


# -------------------------------------------------------------------------
# Seleção das variáveis
# -------------------------------------------------------------------------

st.subheader("2. Seleção das variáveis")

coluna_alvo = st.selectbox(
    "Variável-alvo que deverá ser prevista",
    options=dados.columns,
    index=len(dados.columns) - 1,
)

colunas_disponiveis = [
    coluna for coluna in dados.columns
    if coluna != coluna_alvo
]

variaveis_entrada = st.multiselect(
    "Variáveis de entrada",
    options=colunas_disponiveis,
    default=colunas_disponiveis,
)

if not variaveis_entrada:
    st.warning("Selecione pelo menos uma variável de entrada.")
    st.stop()


# -------------------------------------------------------------------------
# Configuração da rede
# -------------------------------------------------------------------------

st.subheader("3. Configuração da rede neural")

st.caption(
    "Os valores iniciais apresentados são os padrões do MLPRegressor."
)

coluna_1, coluna_2 = st.columns(2)

with coluna_1:
    loss = st.selectbox(
        "Função de perda",
        ["squared_error", "poisson"],
        help=(
            "Poisson é indicado para alvos não negativos, "
            "como contagens ou quantidades."
        ),
    )

    activation = st.selectbox(
        "Função de ativação",
        ["identity", "logistic", "tanh", "relu"],
        index=3,
    )

    solver = st.selectbox(
        "Otimizador",
        ["adam", "sgd", "lbfgs"],
    )

    camadas_texto = st.text_input(
        "Camadas ocultas",
        value="100",
        help=(
            "Informe a quantidade de neurônios separada por vírgulas. "
            "Exemplo: 100, 50 representa duas camadas."
        ),
    )

with coluna_2:
    alpha = st.number_input(
        "Alpha, regularização L2",
        min_value=0.0,
        value=0.0001,
        format="%.6f",
    )

    max_iter = st.number_input(
        "Número máximo de iterações",
        min_value=1,
        value=200,
        step=50,
    )

    tolerancia = st.number_input(
        "Tolerância",
        min_value=0.00000001,
        value=0.0001,
        format="%.8f",
    )

    random_state = st.number_input(
        "Semente aleatória",
        min_value=0,
        value=42,
        step=1,
    )


# -------------------------------------------------------------------------
# Parâmetros condicionais por otimizador
# -------------------------------------------------------------------------

parametros_especificos = {}

with st.expander("Parâmetros avançados"):
    if solver in ["adam", "sgd"]:
        coluna_a, coluna_b = st.columns(2)

        with coluna_a:
            batch_size = st.selectbox(
                "Tamanho do lote",
                ["auto", 16, 32, 64, 128, 256],
            )

            learning_rate_init = st.number_input(
                "Taxa de aprendizado inicial",
                min_value=0.00000001,
                value=0.001,
                format="%.8f",
            )

            shuffle = st.checkbox(
                "Embaralhar registros",
                value=True,
            )

        with coluna_b:
            early_stopping = st.checkbox(
                "Interrupção antecipada",
                value=False,
            )

            n_iter_no_change = st.number_input(
                "Iterações sem melhoria",
                min_value=1,
                value=10,
                step=1,
            )

            validation_fraction = 0.1

            if early_stopping:
                validation_fraction = st.slider(
                    "Proporção para validação",
                    min_value=0.05,
                    max_value=0.40,
                    value=0.10,
                    step=0.05,
                )

        parametros_especificos.update(
            {
                "batch_size": batch_size,
                "learning_rate_init": learning_rate_init,
                "shuffle": shuffle,
                "early_stopping": early_stopping,
                "n_iter_no_change": int(n_iter_no_change),
                "validation_fraction": validation_fraction,
            }
        )

    if solver == "sgd":
        st.markdown("#### Parâmetros do SGD")

        coluna_a, coluna_b = st.columns(2)

        with coluna_a:
            learning_rate = st.selectbox(
                "Estratégia da taxa de aprendizado",
                ["constant", "invscaling", "adaptive"],
            )

            momentum = st.slider(
                "Momentum",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
            )

        with coluna_b:
            nesterovs_momentum = st.checkbox(
                "Utilizar momentum de Nesterov",
                value=True,
            )

            power_t = 0.5

            if learning_rate == "invscaling":
                power_t = st.number_input(
                    "Power T",
                    min_value=0.01,
                    value=0.5,
                    step=0.05,
                )

        parametros_especificos.update(
            {
                "learning_rate": learning_rate,
                "momentum": momentum,
                "nesterovs_momentum": nesterovs_momentum,
                "power_t": power_t,
            }
        )

    if solver == "adam":
        st.markdown("#### Parâmetros do Adam")

        coluna_a, coluna_b, coluna_c = st.columns(3)

        with coluna_a:
            beta_1 = st.number_input(
                "Beta 1",
                min_value=0.0,
                max_value=0.999999,
                value=0.9,
                format="%.6f",
            )

        with coluna_b:
            beta_2 = st.number_input(
                "Beta 2",
                min_value=0.0,
                max_value=0.999999,
                value=0.999,
                format="%.6f",
            )

        with coluna_c:
            epsilon = st.number_input(
                "Epsilon",
                min_value=0.0000000001,
                value=0.00000001,
                format="%.10f",
            )

        parametros_especificos.update(
            {
                "beta_1": beta_1,
                "beta_2": beta_2,
                "epsilon": epsilon,
            }
        )

    if solver == "lbfgs":
        max_fun = st.number_input(
            "Máximo de chamadas da função",
            min_value=1,
            value=15000,
            step=1000,
        )

        parametros_especificos["max_fun"] = int(max_fun)


# -------------------------------------------------------------------------
# Divisão dos dados
# -------------------------------------------------------------------------

st.subheader("4. Divisão dos dados")

test_size = st.slider(
    "Percentual destinado ao teste",
    min_value=0.10,
    max_value=0.50,
    value=0.20,
    step=0.05,
)


# -------------------------------------------------------------------------
# Treinamento
# -------------------------------------------------------------------------

treinar = st.button(
    "🚀 Treinar rede neural",
    type="primary",
    use_container_width=True,
)

if treinar:
    try:
        hidden_layer_sizes = converter_camadas(camadas_texto)
    except ValueError as erro:
        st.error(str(erro))
        st.stop()

    dados_modelo = dados[
        variaveis_entrada + [coluna_alvo]
    ].copy()

    # Garante que a variável-alvo seja numérica.
    dados_modelo[coluna_alvo] = pd.to_numeric(
        dados_modelo[coluna_alvo],
        errors="coerce",
    )

    quantidade_antes = len(dados_modelo)

    dados_modelo = dados_modelo.dropna(
        subset=[coluna_alvo]
    )

    removidos = quantidade_antes - len(dados_modelo)

    if removidos > 0:
        st.warning(
            f"{removidos} registros foram removidos porque a "
            "variável-alvo estava vazia ou não era numérica."
        )

    if len(dados_modelo) < 10:
        st.error(
            "Existem poucos registros válidos para realizar "
            "o treinamento e a avaliação."
        )
        st.stop()

    X = dados_modelo[variaveis_entrada]
    y = dados_modelo[coluna_alvo]

    if loss == "poisson" and (y < 0).any():
        st.error(
            "A função de perda Poisson exige que todos os valores "
            "da variável-alvo sejam maiores ou iguais a zero."
        )
        st.stop()

    colunas_numericas = X.select_dtypes(
        include="number"
    ).columns.tolist()

    colunas_categoricas = [
        coluna for coluna in X.columns
        if coluna not in colunas_numericas
    ]

    pipeline_numerico = Pipeline(
        steps=[
            (
                "preenchimento",
                SimpleImputer(strategy="median"),
            ),
            (
                "padronizacao",
                StandardScaler(),
            ),
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            (
                "preenchimento",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "codificacao",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    transformadores = []

    if colunas_numericas:
        transformadores.append(
            (
                "numericas",
                pipeline_numerico,
                colunas_numericas,
            )
        )

    if colunas_categoricas:
        transformadores.append(
            (
                "categoricas",
                pipeline_categorico,
                colunas_categoricas,
            )
        )

    preprocessador = ColumnTransformer(
        transformers=transformadores,
        remainder="drop",
    )

    parametros_modelo = {
        "loss": loss,
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "alpha": float(alpha),
        "max_iter": int(max_iter),
        "tol": float(tolerancia),
        "random_state": int(random_state),
        **parametros_especificos,
    }

    modelo = MLPRegressor(**parametros_modelo)

    pipeline_completo = Pipeline(
        steps=[
            (
                "preprocessamento",
                preprocessador,
            ),
            (
                "modelo",
                modelo,
            ),
        ]
    )

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
    )

    with st.spinner("Treinando a rede neural..."):
        try:
            with warnings.catch_warnings(record=True) as alertas:
                warnings.simplefilter(
                    "always",
                    ConvergenceWarning,
                )

                pipeline_completo.fit(
                    X_treino,
                    y_treino,
                )

                alertas_convergencia = [
                    alerta
                    for alerta in alertas
                    if issubclass(
                        alerta.category,
                        ConvergenceWarning,
                    )
                ]

            previsoes = pipeline_completo.predict(
                X_teste
            )

        except Exception as erro:
            st.error(
                f"Ocorreu uma falha durante o treinamento: {erro}"
            )
            st.stop()

    if alertas_convergencia:
        st.warning(
            "O modelo atingiu o limite de iterações antes da "
            "convergência. Considere aumentar max_iter, ajustar "
            "a arquitetura ou alterar a taxa de aprendizado."
        )

    mae = mean_absolute_error(
        y_teste,
        previsoes,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_teste,
            previsoes,
        )
    )

    r2 = r2_score(
        y_teste,
        previsoes,
    )

    modelo_treinado = pipeline_completo.named_steps["modelo"]

    st.success("Treinamento concluído.")

    st.subheader("5. Resultados")

    metrica_1, metrica_2, metrica_3, metrica_4 = st.columns(4)

    metrica_1.metric(
        "MAE",
        f"{mae:.6f}",
    )

    metrica_2.metric(
        "RMSE",
        f"{rmse:.6f}",
    )

    metrica_3.metric(
        "R²",
        f"{r2:.6f}",
    )

    metrica_4.metric(
        "Iterações realizadas",
        modelo_treinado.n_iter_,
    )

    resultados = pd.DataFrame(
        {
            "valor_real": y_teste.to_numpy(),
            "valor_previsto": previsoes,
        },
        index=y_teste.index,
    )

    resultados["erro"] = (
        resultados["valor_real"]
        - resultados["valor_previsto"]
    )

    st.markdown("#### Valores reais e previstos")

    st.dataframe(
        resultados,
        use_container_width=True,
    )

    if hasattr(modelo_treinado, "loss_curve_"):
        curva_perda = pd.DataFrame(
            {
                "Perda": modelo_treinado.loss_curve_
            }
        )

        st.markdown("#### Curva de perda")

        st.line_chart(curva_perda)

    st.markdown("#### Configuração utilizada")

    st.json(
        {
            **parametros_modelo,
            "hidden_layer_sizes": list(
                hidden_layer_sizes
            ),
            "variavel_alvo": coluna_alvo,
            "variaveis_entrada": variaveis_entrada,
            "percentual_teste": test_size,
        }
    )

    csv_resultados = resultados.to_csv(
        index=True
    ).encode("utf-8")

    st.download_button(
        "Baixar previsões em CSV",
        data=csv_resultados,
        file_name="previsoes_mlp.csv",
        mime="text/csv",
    )

    arquivo_modelo = io.BytesIO()

    joblib.dump(
        pipeline_completo,
        arquivo_modelo,
    )

    st.download_button(
        "Baixar modelo treinado",
        data=arquivo_modelo.getvalue(),
        file_name="modelo_mlp.joblib",
        mime="application/octet-stream",
    )

    relatorio = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "iteracoes": int(modelo_treinado.n_iter_),
        "parametros": {
            **parametros_modelo,
            "hidden_layer_sizes": list(
                hidden_layer_sizes
            ),
        },
    }

    st.download_button(
        "Baixar relatório JSON",
        data=json.dumps(
            relatorio,
            indent=4,
            ensure_ascii=False,
        ),
        file_name="relatorio_treinamento.json",
        mime="application/json",
    )