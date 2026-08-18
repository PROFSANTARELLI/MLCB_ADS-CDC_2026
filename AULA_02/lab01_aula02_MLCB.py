# ============================================================
# LAB 01 - AULA 02 (MLCB): Classificador de Intenções
# ============================================================
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Dataset de Exemplo (Intenções Bancárias)
dados = {
    'mensagem': [
        'Qual é o saldo da minha conta?',
        'Gostaria de ver meu saldo disponível',
        'Quero fazer um pix para um amigo',
        'Como realizar uma transferência via pix?',
        'Preciso de ajuda para cancelar o cartão',
        'Como faço para efetuar o cancelamento da conta?',
        'Verificar saldo do banco',
        'Transferir dinheiro pelo pix agora'
    ],
    'intencao': [
        'consultar_saldo', 'consultar_saldo',
        'fazer_pix', 'fazer_pix',
        'cancelar_conta', 'cancelar_conta',
        'consultar_saldo', 'fazer_pix'
    ]
}

df1 = pd.DataFrame(dados)

# 2. Separação de Atributos (X) e Rótulos (y)
X = df1['mensagem']
y = df1['intencao']

# 3. Divisão do Dataset em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Vetorização de Texto (Bag of Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Treinamento do Modelo (Regressão Logística)
modelo = LogisticRegression()
modelo.fit(X_train_vec, y_train)

# 6. Teste com Mensagens Inéditas
novas_frases = [
    "Quero consultar quanto dinheiro tenho",
    "Pode me ajudar a fazer um pix?",
    "Gostaria de cancelar meu cartão de crédito"
]

novas_frases_vec = vectorizer.transform(novas_frases)
predicoes = modelo.predict(novas_frases_vec)

print("--- RESULTADOS DO LAB 01 ---")
for frase, intencao in zip(novas_frases, predicoes):
    print(f"Mensagem: '{frase}' ==> Intenção Predita: [{intencao}]")
  

#========== PRODUÇÃO DO RELATÓRIO:==============
# Para a entrega completa deste LAB01 você precisa copiar a saída do código (output) e adicionar as repostas das perguntas abaixo:
# 1 - Avaliem os resultados e verifiquem se os resultados foram corretos ou incorretos. Coloque a resposta no arquivo do relatório do laboratório
# 2 - Detectado algum erro, qual seria a maneira mais correta de melhorar o resultado do algoritmo?
# 3 - Detalhe a função do LogisticRegression no algorítmo.

# Todos os resultados devem ser inseridos no arquivo resultados_aula02.md

#========== FIM ==============
