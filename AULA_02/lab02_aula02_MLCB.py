# ============================================================
# LAB 02 - AULA 02 (MLCB): Naive Bayes e Probabilidades
# ============================================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset de Atendimento de E-Commerce
dados_ecommerce = {
    'mensagem': [
        'Onde esta meu pedido?', 'Qual o status da entrega?', 'Quero rastrear minha compra',
        'Como solicitar reembolso?', 'Quero devolver este produto', 'Preciso do meu dinheiro de volta',
        'Qual o prazo de entrega para SP?', 'Demora quanto tempo para chegar?', 'Qual o frete para o meu CEP?'
    ],
    'intencao': [
        'rastrear_pedido', 'rastrear_pedido', 'rastrear_pedido',
        'troca_devolucao', 'troca_devolucao', 'troca_devolucao',
        'duvida_frete', 'duvida_frete', 'duvida_frete'
    ]
}

df2 = pd.DataFrame(dados_ecommerce)

# Vetorização avançada via TF-IDF
tfidf = TfidfVectorizer()
X_vec = tfidf.fit_transform(df2['mensagem'])
y = df2['intencao']

# Treinamento com Naive Bayes
modelo_nb = MultinomialNB()
modelo_nb.fit(X_vec, y)

# Teste e análise de probabilidade
mensagem_teste = ["Gostaria de devolver o produto que comprei"]
mensagem_vec = tfidf.transform(mensagem_teste)

predicao = modelo_nb.predict(mensagem_vec)[0]
probabilidades = modelo_nb.predict_proba(mensagem_vec)[0]
classes = modelo_nb.classes_

print("--- RESULTADOS DO LAB 02 ---")
print(f"Mensagem de Teste: '{mensagem_teste[0]}'")
print(f"Intenção Predita: {predicao}\n")
print("--- Distribuição de Probabilidades por Classe ---")
for classe, prob in zip(classes, probabilidades):
    print(f"Classe [{classe}]: {prob * 100:.2f}%")

#========== PRODUÇÃO DO RELATÓRIO:==============
# Para a entrega completa deste LAB02 você precisa copiar a saída do código (output) e adicionar as repostas das perguntas abaixo:
# 1 - Avaliem os resultados e verifiquem se os resultados foram corretos ou incorretos. Coloque a resposta no arquivo do relatório do laboratório
# 2 - Detectado algum erro, qual seria a maneira mais correta de melhorar o resultado do algoritmo?
# 3 - Detalhe a função do Naive Bayes no algorítmo.

# Todos os resultados devem ser inseridos no arquivo resultados_aula02.md

#========== FIM ==============
