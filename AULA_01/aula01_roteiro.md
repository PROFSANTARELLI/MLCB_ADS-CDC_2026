Laboratório Prático: Aula 1

O Problema de Negócio

Imagine que uma empresa de e-commerce recebe milhares de mensagens de suporte por dia. Atendentes humanos gastam horas lendo frases simples apenas para descobrir se o cliente está elogiando o serviço ou reclamando de um problema.

A Solução Teórica

Nesta prática, vamos construir um Classificador Supervisionado de Sentimentos simples usando Machine Learning.

[ Entrada: Texto ] ──> [ Vetorizador (Bag of Words) ] ──> [ Classificador Naive Bayes ] ──> [ Saída: Positivo/Negativo ]

Entrada de Texto: Frases escritas em linguagem natural.

Vetorização (CountVectorizer): O computador não entende palavras diretamente; ele entende números. Converteremos cada frase em uma matriz de contagem de palavras (vetor).

Treinamento do Modelo (Multinomial Naive Bayes): Algoritmo probabilístico baseado no Teorema de Bayes que calcula a probabilidade de uma palavra pertencer à classe "positivo" ou "negativo".

Predição: O modelo treinado receberá frases inéditas e dirá automaticamente qual é o sentimento predominante.

Montar seu Notebook Colab a partir do repositório oficial da aula disponibilizado pelo professor no link abaixo:
https://github.com/PROFSANTARELLI/MLCB_ADS-CDC_2026 
