# Construção do modelo de aprendizado supervisionado com todos os aprendizados da aula 1 à 5 (preparatório para prova)
# Objetivo: consolidar todo o pipeline de Natural Language Understanding (NLU) construído até aqui.

#BLOCO 1: Preparação do Ambiente e Carga do Dataset

#Instalos as bibliotecas necessárias para este algoritmo. Fazer em janela separada ou em linha de comando
!pip install gensim
!python -m spacy download pt_core_news_sm

#Importamos as ferramentas de processamento e definimos nosso dataset focado no atendimento imobiliário (vendas, aluguel, suporte e contratos).
import re
import numpy as np
import pandas as pd
import spacy #analisador gramatical - limpa texto, reduz palavra ao seu lema (compraram -> comprar) e identifica a função gramatical de cada termo 
import gensim.downloader as api #baixa e carrega modelos Embeddings pré-treinados (Word2Vec, GloVe e FastText) p/transformar palavras em vetores de significado.
from sklearn.linear_model import LogisticRegression #classe responsável pela matemática propabilística
import gradio as gr #Cria interfaces web interativas e visuais

# 1. Carregamento do modelo de linguagem em português (Spacy)
print("Carregando modelo morfológico Spacy (pt_core_news_sm)")
nlp = spacy.load("pt_core_news_sm")

# 2. Carregamento dos Word Embeddings (Gensim / GloVe de 50 dimensões)
print("Carregando espaço vetorial denso de embeddings modelo Glove")
word_vectors = api.load("glove-wiki-gigaword-50")

# 3. Base de Dados Supervisionada: SAC de Imóveis
dados_imobiliaria = [
    # Intenção: comprar_imovel
    ("Quero comprar um apartamento de 3 quartos com varanda", "comprar_imovel"),
    ("Gostaria de ver casas à venda no centro da cidade", "comprar_imovel"),
    ("Qual o preço médio para compra de cobertura com piscina?", "comprar_imovel"),
    ("Procuro imóvel residencial para comprar com financiamento", "comprar_imovel"),
    ("Vocês têm sobrados à venda na zona sul?", "comprar_imovel"),
    
    # Intenção: alugar_imovel
    ("Procurando kitnet para alugar perto da faculdade", "alugar_imovel"),
    ("Qual o valor do aluguel deste apartamento de 2 dormitórios?", "alugar_imovel"),
    ("Quero alugar um galpão comercial para minha empresa", "alugar_imovel"),
    ("Quais imóveis estão disponíveis para locação imediata?", "alugar_imovel"),
    ("Preciso de uma casa para alugar que aceite animais", "alugar_imovel"),
    
    # Intenção: suporte_manutencao
    ("O chuveiro do apartamento alugado queimou, como pedir conserto?", "suporte_manutencao"),
    ("Muro da casa está com infiltração e vazamento de água", "suporte_manutencao"),
    ("Preciso do contato do encanador para reparo na cozinha", "suporte_manutencao"),
    ("A porta da varanda quebrou, quem faz a manutenção?", "suporte_manutencao"),
    ("Vazamento no teto do banheiro precisa de reparo urgente", "suporte_manutencao"),
    
    # Intenção: 2via_boleto_contrato
    ("Como faço para baixar a segunda via do boleto do aluguel?", "2via_boleto_contrato"),
    ("Não recebi o boleto deste mês para pagamento", "2via_boleto_contrato"),
    ("Preciso do informe de rendimentos e cópia do contrato", "2via_boleto_contrato"),
    ("Onde pego o boleto atualizado com o valor do condomínio?", "2via_boleto_contrato"),
    ("Quero solicitar a segunda via do recibo de pagamento", "2via_boleto_contrato")
]

df = pd.DataFrame(dados_imobiliaria, columns=["mensagem", "intencao"])
print(f"Dataset carregado com {len(df)} mensagens divididas em {df['intencao'].nunique()} intenções.")


BLOCO 2: Esteira NLU (Pré-processamento e Vectorization)
Aqui aplicamos as funções técnicas para transformar texto bruto em vetores semânticos densos de 50 dimensões usando a média das palavras (Mean Pooling).

def preprocessar_texto(texto: str) -> str: #texto bruto para higienizado
    """
    Realiza a limpeza e padronização do texto:
    1. Passa para minúsculas
    2. Remove numerações e caracteres especiais via Regex
    3. Remove Stop Words e extrai os lemas das palavras via Spacy
    """
    texto_limpo = texto.lower()
    texto_limpo = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', '', texto_limpo)
    
    doc = nlp(texto_limpo)  #analisa a gramática e gera um objeto (doc) contendo cada palavra enriquecida
    tokens = [  
        token.lemma_ for token in doc #Cria uma lista filtrada contendo apenas a forma base de cada palavra, aplicando 3 regras de descarte:
        if not token.is_stop and not token.is_space and len(token.text) > 1 #Descarta stop words ("de", "para", "um"), espaços duplos ou quebras de linha e letras isoladas que perderam o sentido (ex: "a", "o", "e").
    ]
    return " ".join(tokens) #Pega a lista de lemas resultantes e os junta novamente em uma única string separada por espaços.

def extrair_sentence_embedding(texto_limpo: str, modelo_emb) -> np.ndarray:
    """
    Converte a frase limpa em um vetor denso único usando a média (Mean Pooling)
    dos embeddings de cada palavra presente no vocabulário.
    """
    palavras = texto_limpo.split() #Quebra a frase limpa em uma lista de palavras individuais
    vetores = [modelo_emb[p] for p in palavras if p in modelo_emb] #Percorre palavra por palavra da lista e faz uma checagem de segurança
    
    if len(vetores) == 0: #trava de segurança contra frases vazias ou desconhecidas
        return np.zeros(modelo_emb.vector_size) #retorna vetor nulo
    
    return np.mean(vetores, axis=0) #: Calcula a média matemática de cada uma das 50 dimensões entre os vetores de todas as palavras da frase.
# O axis=0 instrui o NumPy a calcular a média coluna por coluna (ao longo das linhas), e não da matriz inteira.

# Aplicação no Dataset
df['mensagem_limpa'] = df['mensagem'].apply(preprocessar_texto) #Cria nova coluna na tabela para armazenar o resultado do texto higienizado

# construir a matriz de entrada X de dados numéricos que o modelo vai utilizar para o treinamento
X_densos = np.array([
    extrair_sentence_embedding(txt, word_vectors) for txt in df['mensagem_limpa'] 
]) #Percorre cada frase já limpa da tabela e a transforma em um vetor de 50 números usando o GloVe e converte essa lista de vetores em uma matriz 2D do NumPy, onde cada linha é uma mensagem do dataset e cada coluna representa uma dimensão semântica do embedding.

y = df['intencao'].values #extrai o alvo/rotulo de saída ($y$) que o modelo precisa aprender a prever



BLOCO 3: Treinamento do Modelo e Mapeamento de Respostas Padrão
Neste bloco, treinamos o classificador supervisionado (Regressão Logística) e estruturamos o dicionário de respostas de negócio que a imobiliária utilizará para responder aos clientes.


# 1. Treinamento da Regressão Logística
modelo_nlu = LogisticRegression(C=1.0, max_iter=500)
#Cria o classificador de Regressão Logística. C=1.0: Ajusta o nível de regularização (evita que o modelo decore os dados/overfitting). max_iter=500: Define o número máximo de tentativas/iterações para encontrar os pesos matemáticos ideais.

modelo_nlu.fit(X_densos, y) #Conecta as entradas numéricas (X_densos) com as intenções corretas (y).

print("Modelo supervisionado treinado!")

# 2. Base de Conhecimento: Respostas Padrão de Negócio do SAC
RESPOSTAS_PADRAO = {
    "comprar_imovel": (
        "**Atendimento de Vendas:** Ficamos felizes com seu interesse! "
        "Você pode conferir nosso catálogo de imóveis à venda em nosso site www.imobiliaria.com/vendas "
        "ou aguardar que um de nossos corretores entrará em contato em instantes."
    ),
    "alugar_imovel": (
        "**Atendimento de Locação:** Temos ótimas opções disponíveis! "
        "Acesse www.imobiliaria.com/aluguel para filtrar por região e valor. "
        "Para agendar uma visita, envie o código do imóvel por aqui."
    ),
    "suporte_manutencao": (
        "**Suporte e Manutenção:** Sentimos muito pelo inconveniente. "
        "Por favor, abra um chamado urgente em nosso portal do inquilino (www.imobiliaria.com/manutencao) "
        "anexando fotos ou vídeos do problema para acionarmos nossos prestadores."
    ),
    "2via_boleto_contrato": (
        "**Financeiro e Contratos:** Para acessar boletos ou documentos, "
        "acesse a Área do Cliente em www.imobiliaria.com/cliente informando seu CPF e senha. "
        "Lá você baixa a 2ª via atualizada em segundos."
    )
}


BLOCO 4: Motor de Inferência com Lógica de Fallback
Criamos a função principal que recebe uma mensagem nova, limpa, gera o embedding, executa a predição, avalia o nível de confiança e define a resposta final.

LIMIAR_CONFIANCA = 0.50  # Limiar de corte para aceitação da resposta do bot

def processar_atendimento_sac(mensagem_usuario: str):
    if not mensagem_usuario or not mensagem_usuario.strip():
        return "N/A", "0.0%", " Aguardando mensagem...", "Aguardando entrada do usuário..."
        
    # Step 1: Pré-processamento
    msg_limpa = preprocessar_texto(mensagem_usuario)
    
    # Step 2: Vetorização
    vetor_input = extrair_sentence_embedding(msg_limpa, word_vectors).reshape(1, -1)
    
    # Step 3: Predição de Probabilidades
    probabilidades = modelo_nlu.predict_proba(vetor_input)[0]
    idx_maior_prob = np.argmax(probabilidades)
    confianca = probabilidades[idx_maior_prob]
    intencao_detectada = modelo_nlu.classes_[idx_maior_prob]
    
    percentual_confianca = f"{confianca * 100:.1f}%"
    
    # Step 4: Regra de Decisão / Fallback
    if confianca >= LIMIAR_CONFIANCA:
        classificacao_status = f" IDENTIFICADO ({intencao_detectada})"
        texto_resposta = RESPOSTAS_PADRAO[intencao_detectada]
    else:
        classificacao_status = " UNCERTAIN (Fallback Acionado)"
        texto_resposta = (
            "Desculpe, não consegui compreender com clareza a sua solicitação. "
            "Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento."
        )
    
    # Formata a resposta com um Box estilizado em HTML/Markdown
    card_resposta = f"""
    <div style="background-color: #f0f4f9; border-left: 5px solid #2b5c8f; padding: 15px; border-radius: 8px; margin-top: 10px;">
        <h4 style="margin: 0 0 8px 0; color: #2b5c8f;"> Resposta Automática do SAC:</h4>
        <p style="margin: 0; font-size: 15px; color: #1a1a1a;">{texto_resposta}</p>
    </div>
    """
        
    return intencao_detectada, percentual_confianca, classificacao_status, card_resposta


BLOCO 5: Construção da Interface Gráfica Interativa (Gradio)
Construímos a tela web com campo de input, botão de processamento e exibição de todos os indicadores operacionais em tempo real.

import gradio as gr

with gr.Blocks(theme=gr.themes.Soft(), title="SAC Imobiliário") as app:
    gr.Markdown(
        """
        #  SAC Imobiliário — ChatBot Inteligente
        *Será um prazer atendê-lo. Digite a seguir a sua necessidade*
        """
    )
    
    with gr.Row():
        # COLUNA DA ESQUERDA: Entrada de Dados
        with gr.Column(scale=1):
            gr.Markdown("###  Mensagem do Cliente")
            input_texto = gr.Textbox(
                lines=4, 
                placeholder="Ex: Preciso da segunda via do boleto de aluguel...",
                label="Digite sua necessidade"
            )
            btn_processar = gr.Button(" Processar Mensagem", variant="primary", size="lg")
            
            # Exemplos práticos rápidos para os alunos clicarem durante a aula
            gr.Examples(
                examples=[
                    ["Preciso de suporte técnico para consertar vazamento."],
                    ["Quero ver apartamentos à venda na zona sul."],
                    ["Como faço para alugar um galpão comercial?"],
                    ["Gostaria de baixar o boleto do condomínio."],
                    ["Vocês vendem terreno na Lua ou em Marte?"] # Exemplo de Fallback
                ],
                inputs=input_texto
            )
        
        # COLUNA DA DIREITA: Indicadores Técnicos + Resposta Formatada
        with gr.Column(scale=1):
            gr.Markdown("### Painel de Diagnóstico")
            
            # Indicadores numéricos topo a topo
            with gr.Row():
                out_intencao = gr.Textbox(label="Intenção", scale=2, interactive=False)
                out_confianca = gr.Textbox(label="Confiança", scale=1, interactive=False)
            
            out_status = gr.Textbox(label="Status da Decisão", interactive=False)
            
            # Resposta visual envelopada no card estilizado
            out_resposta = gr.HTML(
                value="<div style='padding: 15px; color: #888;'>Aguardando envio de mensagem...</div>", 
                label="Resposta da Imobiliária"
            )

    # Ação do Botão
    btn_processar.click(
        fn=processar_atendimento_sac,
        inputs=[input_texto],
        outputs=[out_intencao, out_confianca, out_status, out_resposta]
    )

# Executar a aplicação
app.launch(debug=True, share=True)
#FIM DO CÓDIGO
