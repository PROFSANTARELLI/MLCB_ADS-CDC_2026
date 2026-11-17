# Roteiro de Estudo — Inteligência Artificial, NLP e Sistemas Interativos

1 - Foquem estudos nas aulas de 5 à 12.

2 - Refaçam atividades e tentem aprender de verdade

3 - Use a IA como fonte de estudo apenas. Peça para ela te ajudar a estudar.

# Sobre a prova:
1 - Serão 3 questões técnicas com valor de 1 ponto cada

2 - A prova é sem consulta e não será permitido uso de IA


## 1. Processamento de Linguagem Natural (NLP) e Classificação de Intenções

Modelos matemáticos não processam texto puro diretamente. Para classificar uma frase (como identificar a intenção de um usuário), o texto passa por um pipeline de transformação numérica e inferência probabilística.

### 1.1. Vetorização de Texto: Bag of Words vs. TF-IDF
* **Bag of Words (BoW):** Constrói um vocabulário com todas as palavras únicas do conjunto de treino e conta quantas vezes cada termo aparece em cada frase. Sua limitação é dar o mesmo peso a palavras muito comuns (ex.: *"para"*, *"com"*, *"hoje"*) e palavras altamente específicas.
* **TF-IDF (*Term Frequency - Inverse Document Frequency*):** Corrige a limitação do BoW multiplicando a frequência do termo na frase (**TF**) pelo inverso da sua frequência em todo o conjunto de documentos (**IDF**):
  $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$
  * **Efeito prático:** Termos que aparecem em quase todas as frases do dataset têm seu peso penalizado (próximo de zero), enquanto palavras raras e discriminativas recebem pesos elevados.

### 1.2. A Regra de Ouro: `.fit_transform()` vs. `.transform()`
* **No Treinamento (`.fit_transform(X_treino)`):** O vetorizador aprende o vocabulário do dataset (define quais colunas existem na matriz) **e** converte os textos em números.
* **Na Inferência/Produção (`.transform([texto_novo])`):** O vetorizador **apenas converte** a nova frase usando as colunas já fixadas no treino.
  > **Atenção:** Se `.fit_transform()` for chamado durante a inferência em uma única frase nova, o vetorizador recriará o vocabulário apenas com as palavras daquela frase. Isso altera o número de colunas da matriz de entrada e provoca erro fatal de dimensionalidade (`ValueError`) ao passar os dados para o modelo treinado.

### 1.3. Trava de Segurança (*Threshold*) vs. `.predict()`
* Usar `modelo.predict(X)` força o classificador a sempre escolher a classe com a maior probabilidade, mesmo que essa probabilidade seja baixíssima (ex.: `[0.25, 0.28, 0.24, 0.23]` em um problema de 4 classes).
* **Solução Arquitetural:** Utilizar `modelo.predict_proba(X)[0]`, extrair a maior probabilidade com `np.max()` (ou `np.argmax()`) e compará-la a um **limiar de confiança (*threshold*)** (ex.: `0.65`). Caso a confiança seja menor que o limiar, o sistema aciona uma resposta de contingência (*fallback*), evitando falsos positivos.

### 1.4. Desbalanceamento de Classes e Métricas de Avaliação
* Quando um dataset possui 95% de exemplos na Classe A e apenas 2% na Classe C, a **Acurácia Global** torna-se enganosa (um modelo que chuta sempre "A" atinge 95% de acurácia sem aprender nada sobre "C").
* **Como diagnosticar:** Analisar a **Matriz de Confusão** e avaliar métricas por classe:
  * **Recall (Revocação):** De todos os exemplos reais da Classe C, quantos o modelo conseguiu encontrar?
  * **Precision (Precisão):** Das vezes em que o modelo previu Classe C, quantas ele realmente acertou?
  * **F1-Score:** Média harmônica entre *Precision* e *Recall*, ideal para avaliar classes minoritárias.

---

## 2. Extração Determinística de Entidades com Expressões Regulares (Regex)

Enquanto modelos de Machine Learning classificam a intenção geral (categorias discretas), eles não extraem substrings variáveis (números, códigos, datas) com precisão determinística. Para isso, utiliza-se o módulo nativo `re` do Python.

### 2.1. Anatomia de um Padrão Regex
Considere o padrão: `r'(\d+(?:[\.,]\d+)?)\s*(?:kg|quilos|g)\b'`
* `\d+`: Um ou mais dígitos numéricos (`0-9`).
* `(...)`: **Grupo de Captura**. Tudo que casar dentro destes parênteses fica salvo em memória e pode ser acessado via `match.group(1)`.
* `(?:...)`: **Grupo de Não-Captura (*Non-Capturing Group*)**. Serve apenas para agrupar alternativas (`|`) ou aplicar quantificadores (`?`, `+`) sem criar um índice de grupo extra. Se o `?:` for removido, os parênteses passam a ser contados como grupos de captura, deslocando a numeração dos grupos subsequentes.
* `[\.,]\d+`: Um ponto literal ou vírgula literal seguido de um ou mais dígitos (para números decimais). O `?` logo após o grupo indica que a parte decimal é opcional.
* `\s*`: Zero ou mais espaços em branco entre o número e a unidade.
* `\b`: Borda de palavra (*word boundary*), garantindo que `g` não case com o início da palavra `"gramas"` ou `"garrafa"` se não for a unidade isolada.
* `(?!\d)`: **Lookahead Negativo**. Garante que o padrão não seja seguido imediatamente por outro dígito (ex.: `r'^[A-Z]{2,4}-\d{3,5}(?!\d)'` aceita `XPTO-4501`, mas rejeita `XYZ-987654` porque há um sexto dígito após os 5 primeiros).

### 2.2. Comportamento de `re.search()` vs. `re.findall()`
* **`re.search(padrao, texto, flags)`:** Varre o texto da esquerda para a direita e **para na primeira ocorrência encontrada**, retornando um objeto `Match` (ou `None` se nada casar). Se houver duas medidas na mesma frase (ex.: `"500g de café e 2 kg de açúcar"`), apenas `"500g"` será capturada.
* **Normalização Pós-Captura:** Números extraídos de textos em português frequentemente usam vírgula decimal (`"12,50"`). Antes de converter para ponto flutuante (`float()`), é obrigatório aplicar `.replace(',', '.')`.

---

## 3. Aprendizado Supervisionado vs. Busca Baseada em Instâncias

### 3.1. Random Forest (*Ensemble* por *Bagging*)
* É um comitê de múltiplas Árvores de Decisão treinadas de forma independente e paralela.
* **Mecanismo de Redução de Variância:** Utiliza duas fontes de aleatoriedade:
  1. **Bootstrap Aggregating (*Bagging*):** Cada árvore é treinada em uma amostra aleatória *com reposição* dos dados originais.
  2. **Subespaços Aleatórios de Atributos:** Em cada divisão de nó, a árvore avalia apenas um subconjunto aleatório das *features* disponíveis, evitando que uma única variável dominante torne todas as árvores idênticas.
* Por ser baseado em regras de corte ortogonais (eixos), **não exige normalização de escala** das variáveis numéricas.

### 3.2. K-Nearest Neighbors (KNN) e Espaços Vetoriais
* **Lazy Learner (Aprendizado Preguiçoso):** Durante o método `.fit(X)`, o KNN não ajusta pesos nem constrói equações; ele apenas armazena os pontos na memória. Todo o custo computacional ocorre na fase de **inferência** (`.kneighbors()` ou `.predict()`), apresentando complexidade linear em relação ao tamanho da base ($O(N \cdot D)$), o que exige estruturas de indexação (como *KD-Trees*) quando $N$ chega a milhões de registros.
* **Distância Euclidiana:** Mede a linha reta entre dois pontos $p = (x_1, y_1)$ e $q = (x_2, y_2)$ no espaço $n$-dimensional:
  $$d(p, q) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
* **Impacto do Hiperparâmetro $K$ e da Escala:**
  * $K$ muito pequeno ($K=1$): Fronteira de decisão altamente irregular e sensível a ruídos (*overfitting* / alta variância).
  * $K$ muito grande: Suaviza excessivamente a fronteira, misturando regiões distantes (*underfitting* / alto viés).
  * **Sensibilidade à Escala:** Variáveis com magnitudes maiores (ex.: salário em milhares vs. idade em dezenas) dominam o cálculo da distância se não forem previamente normalizadas (*MinMaxScaler* ou *StandardScaler*).
