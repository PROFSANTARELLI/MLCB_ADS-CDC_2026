# ============================================================
# LAB 01 - AULA 03 (MLCB): Pré-processamento e Stopwords
# ============================================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dataset de Atendimento Financeiro
dados = {
    'mensagem': [
        'Como posso emitir a segunda via do meu boleto?',
        'Preciso da 2a via da minha fatura atrasada',
        'Quero negociar o pagamento da minha dívida',
        'Como fazer um acordo para pagar o débito em aberto?',
        'Gostaria de alterar meu endereço de cadastramento',
        'Onde atualizo os meus dados residenciais no app?'
    ],
    'intencao': [
        'segunda_via', 'segunda_via',
        'negociar_divida', 'negociar_divida',
        'atualizar_cadastro', 'atualizar_cadastro'
    ]
}

df1 = pd.DataFrame(dados)

# Criando lista de Stopwords personalizadas em Português
stopwords_pt = [
    'de', 'da', 'do', 'dos', 'das', 'a', 'o', 'as', 'os', 'em', 'para',
    'com', 'por', 'meu', 'minha', 'meus', 'minhas', 'como', 'quero', 'preciso'
]

# Vetorização TF-IDF aplicando stopwords e n-grams
vectorizer = TfidfVectorizer(stop_words=stopwords_pt, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(df1['mensagem'])
y = df1['intencao']

modelo = LogisticRegression()
modelo.fit(X_vec, y)

# Teste com nova mensagem
frase_teste = ["Preciso urgente da segunda via da fatura"]
frase_vec = vectorizer.transform(frase_teste)
predicao = modelo.predict(frase_vec)[0]

print("--- RESULTADOS DO LAB 01 (AULA 03) ---")
print(f"Mensagem: '{frase_teste[0]}'")
print(f"Intenção Predita: [{predicao}]")
print(f"Vocabulário Filtrado (sem stopwords): {list(vectorizer.get_feature_names_out())}")

#========== PRODUÇÃO DO RELATÓRIO:==============
# 1 - Qual o impacto da remoção de stopwords no tamanho do vocabulário do modelo?
# 2 - O que significa a configuração ngram_range=(1, 2) no TfidfVectorizer?
# 3 - Como a remoção de palavras genéricas ajuda a evitar classificações incorretas?
# Todos os resultados devem ser inseridos no arquivo resultados_aula03.md
#========== FIM ==============

