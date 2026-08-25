# ============================================================
# LAB 03 - AULA 03 (MLCB): Scikit-Learn Pipeline (Modo TODO)
# ============================================================
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dados_rh = {
    'mensagem': [
        'Como solicitar minhas ferias?', 'Quero agendar meu periodo de ferias',
        'Onde baixo meu holerite do mes?', 'Preciso do comprovante de rendimentos',
        'Como cadastrar meu atestado medico?', 'Onde envio o atestado de consulta?'
    ],
    'intencao': [
        'solicitar_ferias', 'solicitar_ferias',
        'obter_holerite', 'obter_holerite',
        'enviar_atestado', 'enviar_atestado'
    ]
}

df3 = pd.DataFrame(dados_rh)

# TODO 1: Separe o dataset em X ('mensagem') e y ('intencao')
X = None
y = None

# TODO 2: Realize o train_test_split com test_size=0.33 e random_state=42
# X_train, X_test, y_train, y_test = ...

# TODO 3: Monte o Pipeline encapsulando o TfidfVectorizer e a LogisticRegression
# pipeline = Pipeline([
#     ('vectorizer', TfidfVectorizer(stop_words=['de', 'o', 'meu', 'minhas'])),
#     ('classifier', LogisticRegression())
# ])

# TODO 4: Treine o pipeline completo com .fit() usando os dados de treino brutos
# pipeline.fit(...)

# TODO 5: Faca a predicao nos dados de teste brutos e exiba a acuracia
# predicoes = pipeline.predict(...)
# print(f"Acuracia via Pipeline: {accuracy_score(y_test, predicoes) * 100:.2f}%")

#========== PRODUÇÃO DO RELATÓRIO:==============
# 1 - Cole o código corrigido e a acurácia obtida.
# 2 - Qual é a grande vantagem de utilizar o objeto Pipeline no Scikit-Learn?
# 3 - Por que o Pipeline evita que erros de pré-processamento ocorram entre treino e teste?
# Todos os resultados devem ser inseridos no arquivo resultados_aula03.md
#========== FIM ==============
