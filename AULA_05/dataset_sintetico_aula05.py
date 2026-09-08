import pandas as pd


# ============================================================
# DATASET DE TREINAMENTO
# ============================================================

dados_treino = [

    # --------------------------------------------------------
    # TROCAS E DEVOLUÇÕES
    # --------------------------------------------------------

    ("Quero devolver este sofá que chegou com rasgo",
     "trocas_devolucoes"),

    ("Gostaria de trocar minha mesa que veio arranhada",
     "trocas_devolucoes"),

    ("Como faço para solicitar a devolução do meu estofado?",
     "trocas_devolucoes"),

    ("O rack veio com defeito e quero trocar",
     "trocas_devolucoes"),

    ("Preciso devolver a cadeira de escritório com defeito",
     "trocas_devolucoes"),

    ("Quero cancelar a compra e pedir estorno do sofá",
     "trocas_devolucoes"),

    ("Meu painel de TV veio quebrado e quero trocar",
     "trocas_devolucoes"),

    ("Gostaria de devolver o armário por defeito",
     "trocas_devolucoes"),

    ("Como faço para trocar um móvel que veio danificado?",
     "trocas_devolucoes"),

    ("Quero solicitar a troca da minha estante",
     "trocas_devolucoes"),

    ("O produto chegou diferente do que comprei",
     "trocas_devolucoes"),

    ("Preciso devolver uma cadeira que apresentou defeito",
     "trocas_devolucoes"),

    ("Meu sofá chegou rasgado, como faço a troca?",
     "trocas_devolucoes"),

    ("Quero devolver o produto que recebi",
     "trocas_devolucoes"),

    ("Posso trocar minha mesa de jantar?",
     "trocas_devolucoes"),

    ("Como funciona o processo de devolução?",
     "trocas_devolucoes"),

    ("Quero cancelar meu pedido e devolver o produto",
     "trocas_devolucoes"),

    ("A cama chegou danificada e preciso trocar",
     "trocas_devolucoes"),

    ("Recebi um armário com peças quebradas e quero devolver",
     "trocas_devolucoes"),

    ("Gostaria de saber como solicitar o estorno",
     "trocas_devolucoes"),


    # --------------------------------------------------------
    # LOGÍSTICA E ENTREGAS
    # --------------------------------------------------------

    ("Qual o status da entrega da minha estante?",
     "logistica_entregas"),

    ("Onde está meu pedido de poltrona?",
     "logistica_entregas"),

    ("Qual o prazo de entrega do sofá que comprei?",
     "logistica_entregas"),

    ("Meu armário de cozinha ainda não chegou",
     "logistica_entregas"),

    ("Quero rastrear o transporte da minha mesa de jantar",
     "logistica_entregas"),

    ("A entrega do guarda roupa está atrasada",
     "logistica_entregas"),

    ("Quando chega minha cadeira presidente?",
     "logistica_entregas"),

    ("Quero saber o dia que chegam meus móveis",
     "logistica_entregas"),

    ("Como faço para rastrear meu pedido?",
     "logistica_entregas"),

    ("Onde está minha compra?",
     "logistica_entregas"),

    ("Meu pedido está atrasado",
     "logistica_entregas"),

    ("Quero consultar o status da entrega",
     "logistica_entregas"),

    ("Quando meu sofá será entregue?",
     "logistica_entregas"),

    ("Ainda não recebi minha mesa",
     "logistica_entregas"),

    ("Qual a previsão de chegada do meu pedido?",
     "logistica_entregas"),

    ("Preciso acompanhar a entrega do meu armário",
     "logistica_entregas"),

    ("Minha encomenda ainda não chegou",
     "logistica_entregas"),

    ("Tem como verificar onde está meu pedido?",
     "logistica_entregas"),

    ("Gostaria de saber o prazo de entrega",
     "logistica_entregas"),

    ("Meu pedido já foi enviado?",
     "logistica_entregas"),


    # --------------------------------------------------------
    # SUPORTE TÉCNICO
    # --------------------------------------------------------

    ("Como montar o painel de TV da sala?",
     "suporte_tecnico"),

    ("Não consigo entender o manual de montagem do rack",
     "suporte_tecnico"),

    ("Faltaram parafusos no kit do meu guarda roupa",
     "suporte_tecnico"),

    ("Preciso de ajuda para ajustar a porta do armário",
     "suporte_tecnico"),

    ("A peça B da mesa não encaixa na peça C",
     "suporte_tecnico"),

    ("Como regulo a altura da minha cadeira ergonômica?",
     "suporte_tecnico"),

    ("Vocês enviam montador para a estante?",
     "suporte_tecnico"),

    ("O manual da cama de casal veio em branco",
     "suporte_tecnico"),

    ("Preciso de ajuda para montar minha mesa",
     "suporte_tecnico"),

    ("Como faço a montagem do guarda roupa?",
     "suporte_tecnico"),

    ("Não sei como instalar o painel",
     "suporte_tecnico"),

    ("Meu móvel veio sem algumas peças",
     "suporte_tecnico"),

    ("Faltaram parafusos na embalagem",
     "suporte_tecnico"),

    ("Como ajustar a porta do meu armário?",
     "suporte_tecnico"),

    ("Tenho dificuldade para montar o rack",
     "suporte_tecnico"),

    ("A peça do meu móvel não encaixa",
     "suporte_tecnico"),

    ("Preciso do manual de montagem da mesa",
     "suporte_tecnico"),

    ("Como instalar corretamente minha estante?",
     "suporte_tecnico"),

    ("Posso contratar um montador para meu móvel?",
     "suporte_tecnico"),

    ("Meu móvel veio com uma peça faltando",
     "suporte_tecnico"),


    # --------------------------------------------------------
    # VENDAS E ORÇAMENTO
    # --------------------------------------------------------

    ("Qual o valor da mesa de jantar de seis lugares?",
     "vendas_orcamento"),

    ("Gostaria de um orçamento de sofá retrátil",
     "vendas_orcamento"),

    ("Vocês têm desconto para pagamento via pix na poltrona?",
     "vendas_orcamento"),

    ("Quanto custa o frete para o guarda roupa de casal?",
     "vendas_orcamento"),

    ("Tem promoção de cômoda este mês?",
     "vendas_orcamento"),

    ("Qual o preço do armário de cozinha planejado?",
     "vendas_orcamento"),

    ("Gostaria de comprar um beliche de madeira",
     "vendas_orcamento"),

    ("Quais as formas de parcelamento do rack?",
     "vendas_orcamento"),

    ("Quanto custa um sofá de três lugares?",
     "vendas_orcamento"),

    ("Quero saber o preço de uma mesa de jantar",
     "vendas_orcamento"),

    ("Vocês têm promoção de sofá?",
     "vendas_orcamento"),

    ("Quais são as opções de pagamento?",
     "vendas_orcamento"),

    ("É possível parcelar minha compra?",
     "vendas_orcamento"),

    ("Gostaria de receber um orçamento",
     "vendas_orcamento"),

    ("Quero comprar uma cadeira de escritório",
     "vendas_orcamento"),

    ("Quanto custa uma estante?",
     "vendas_orcamento"),

    ("Tem desconto para pagamento à vista?",
     "vendas_orcamento"),

    ("Quais sofás vocês têm disponíveis?",
     "vendas_orcamento"),

    ("Qual o preço de uma cama de casal?",
     "vendas_orcamento"),

    ("Quero saber os preços dos móveis disponíveis",
     "vendas_orcamento"),
]


# ============================================================
# DATASET DE TESTE
#
# Importante:
# Estas frases NÃO devem aparecer no treinamento.
# ============================================================

dados_teste = [

    # Trocas e devoluções
    ("Meu sofá chegou com problema e quero fazer a troca",
     "trocas_devolucoes"),

    ("Como posso devolver uma compra?",
     "trocas_devolucoes"),

    ("Quero trocar o móvel que recebi",
     "trocas_devolucoes"),

    ("Recebi uma mesa danificada",
     "trocas_devolucoes"),

    ("Preciso pedir devolução do produto",
     "trocas_devolucoes"),

    ("Como solicito o estorno da compra?",
     "trocas_devolucoes"),

    ("Quero devolver uma cadeira que comprei",
     "trocas_devolucoes"),

    ("Meu armário veio quebrado",
     "trocas_devolucoes"),


    # Logística
    ("Quero saber onde está minha compra",
     "logistica_entregas"),

    ("Minha entrega ainda não foi realizada",
     "logistica_entregas"),

    ("Quando meu pedido vai chegar?",
     "logistica_entregas"),

    ("Gostaria de acompanhar minha encomenda",
     "logistica_entregas"),

    ("Meu móvel ainda não chegou",
     "logistica_entregas"),

    ("Tem como rastrear minha compra?",
     "logistica_entregas"),

    ("Qual a previsão para minha entrega?",
     "logistica_entregas"),

    ("Meu pedido está demorando muito",
     "logistica_entregas"),


    # Suporte
    ("Como faço para montar meu guarda roupa?",
     "suporte_tecnico"),

    ("Não consigo montar minha mesa",
     "suporte_tecnico"),

    ("Meu móvel veio sem parafusos",
     "suporte_tecnico"),

    ("Preciso do manual de instalação",
     "suporte_tecnico"),

    ("Como encaixar as peças do armário?",
     "suporte_tecnico"),

    ("É possível contratar um montador?",
     "suporte_tecnico"),

    ("Uma peça não encaixa corretamente",
     "suporte_tecnico"),

    ("Como instalar o rack?",
     "suporte_tecnico"),


    # Vendas
    ("Quanto custa um sofá retrátil?",
     "vendas_orcamento"),

    ("Quero fazer um orçamento de uma mesa",
     "vendas_orcamento"),

    ("Vocês aceitam pagamento parcelado?",
     "vendas_orcamento"),

    ("Existe desconto no pagamento à vista?",
     "vendas_orcamento"),

    ("Quero comprar uma poltrona",
     "vendas_orcamento"),

    ("Qual o valor de uma cama?",
     "vendas_orcamento"),

    ("Quais são as formas de pagamento?",
     "vendas_orcamento"),

    ("Vocês têm alguma promoção?",
     "vendas_orcamento"),
]


# ============================================================
# DATASET OOD
#
# Out Of Domain:
# não pertence a nenhuma intenção conhecida.
# ============================================================

dados_ood = [

    ("Qual é a previsão do tempo para amanhã?",
     "fora_do_dominio"),

    ("Quem ganhou o campeonato brasileiro?",
     "fora_do_dominio"),

    ("Qual é o valor do dólar hoje?",
     "fora_do_dominio"),

    ("Me ensine a fazer um bolo de chocolate",
     "fora_do_dominio"),

    ("Qual é a capital da França?",
     "fora_do_dominio"),

    ("Quais filmes estão passando hoje?",
     "fora_do_dominio"),

    ("Preciso marcar uma consulta médica",
     "fora_do_dominio"),

    ("Como faço para declarar meu imposto de renda?",
     "fora_do_dominio"),

    ("Qual será o clima no final de semana?",
     "fora_do_dominio"),

    ("Me conte uma piada",
     "fora_do_dominio"),

    ("Como aprender inglês rapidamente?",
     "fora_do_dominio"),

    ("Qual é a melhor linguagem de programação?",
     "fora_do_dominio"),

    ("Quem inventou a internet?",
     "fora_do_dominio"),

    ("Quero reservar uma passagem aérea",
     "fora_do_dominio"),

    ("Qual restaurante você recomenda?",
     "fora_do_dominio"),

    ("Me explique como funciona Bitcoin",
     "fora_do_dominio"),

    ("Qual é o melhor celular atualmente?",
     "fora_do_dominio"),

    ("Quero aprender a tocar violão",
     "fora_do_dominio"),

    ("Como faço uma receita de lasanha?",
     "fora_do_dominio"),

    ("Qual será o resultado do próximo jogo?",
     "fora_do_dominio"),
]


# ============================================================
# CRIAÇÃO DOS DATAFRAMES
# ============================================================

df_treino = pd.DataFrame(
    dados_treino,
    columns=["mensagem", "intencao"]
)

df_teste = pd.DataFrame(
    dados_teste,
    columns=["mensagem", "intencao"]
)

df_ood = pd.DataFrame(
    dados_ood,
    columns=["mensagem", "intencao"]
)


# ============================================================
# EMBARALHAR OS DATASETS
# ============================================================

df_treino = df_treino.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df_teste = df_teste.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df_ood = df_ood.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SALVAR CSVs
# ============================================================

df_treino.to_csv(
    "sac_moveis_ac2_treino.csv",
    index=False,
    encoding="utf-8"
)

df_teste.to_csv(
    "sac_moveis_ac2_teste.csv",
    index=False,
    encoding="utf-8"
)

df_ood.to_csv(
    "sac_moveis_ac2_ood.csv",
    index=False,
    encoding="utf-8"
)


# ============================================================
# RELATÓRIO
# ============================================================

print("=" * 60)
print("DATASETS DA MÓVEISDESIGN")
print("=" * 60)

print(f"\nTreinamento: {len(df_treino)} mensagens")
print(f"Teste:       {len(df_teste)} mensagens")
print(f"OOD:         {len(df_ood)} mensagens")

print("\nDistribuição do treinamento:")
print(df_treino["intencao"].value_counts())

print("\nDistribuição do teste:")
print(df_teste["intencao"].value_counts())

print("\nArquivos gerados:")
print("✓ sac_moveis_ac2_treino.csv")
print("✓ sac_moveis_ac2_teste.csv")
print("✓ sac_moveis_ac2_ood.csv")
