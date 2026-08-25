# ============================================================
# LAB 02 - AULA 03 (MLCB): Matriz de Confusão e Métricas
# ============================================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Dataset ampliado para avaliação estatística
dados = {
    'mensagem': [
        'Onde fica a loja fisica?', 'Qual o endereço da unidade SP?', 'Como chegar na loja?',
        'Qual o horario de funcionamento?', 'A loja abre aos domingos?', 'Que horas voces fecham?',
        'Quero trocar um produto com defeito', 'Como funciona a troca?', 'Preciso devolver meu pedido'
    ],
    'intencao': [
        'localizacao', 'localizacao', 'localizacao',
        'horario_atendimento', 'horario_atendimento', 'horario_atendimento',
        'troca_devolucao', 'troca_devolucao', 'troca_devolucao'
    ]
}

df2 = pd.DataFrame(dados)

X_train, X_test, y_train, y_test = train_test_split(
    df2['mensagem'], df2['intencao'], test_size=0.33, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

modelo = MultinomialNB()
modelo.fit(X_train_vec, y_train)

y_pred = modelo.predict(X_test_vec)

print("--- RESULTADOS DO LAB 02 (AULA 03) ---")
print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("--- Matriz de Confusão ---")
print(confusion_matrix(y_test, y_pred))

#========== PRODUÇÃO DO RELATÓRIO:==============
# 1 - O que representam as métricas Precision, Recall e F1-Score no relatório?
# 2 - Como interpretar a diagonal principal da Matriz de Confusão?
# 3 - Por que a acurácia isolada pode ser enganosa quando temos classes desbalanceadas?
# Todos os resultados devem ser inseridos no arquivo resultados_aula03.md
#========== FIM ==============
