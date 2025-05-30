# Classificação de Pneumonia em Raios-X de Tórax com CNN e Interface Web

Este projeto implementa uma rede neural convolucional (CNN) para classificar imagens de raios-x de tórax em duas classes: **NORMAL** e **PNEUMONIA**. Além disso, oferece uma interface web simples usando Streamlit para que qualquer usuário possa fazer upload de imagens e receber a classificação.

---

## Estrutura do projeto

- Visualização inicial de imagens do dataset para inspeção
- Pré-processamento e aumento de dados (data augmentation) com `ImageDataGenerator`
- Construção, compilação e treino de uma CNN para classificação binária
- Avaliação do modelo com gráficos, matriz de confusão e relatório de classificação
- Aplicação web simples com Streamlit para teste do modelo com imagens customizadas

---

## Descrição do código

### 1. Visualização de imagens do dataset

- As imagens são carregadas de diretórios locais organizados em:
  - `/content/chest_xray/train/NORMAL`
  - `/content/chest_xray/train/PNEUMONIA`
- 3 imagens são selecionadas aleatoriamente de cada classe e exibidas lado a lado para inspeção visual.

---

### 2. Pré-processamento e geração de dados

- Utiliza `ImageDataGenerator` do Keras para:
  - Normalizar as imagens (dividir pixels por 255)
  - Aplicar aumento de dados (rotação, zoom, deslocamento e flip horizontal) no conjunto de treino para melhorar generalização.
- Os dados são carregados em batches a partir das pastas `train`, `val` (validação) e `test`.

---

### 3. Construção e treinamento da CNN

- Arquitetura da CNN:
  - 3 camadas convolucionais com ReLU e MaxPooling
  - Camada `Flatten`
  - Camada densa de 128 neurônios com dropout de 50% para evitar overfitting
  - Saída sigmoid para classificação binária
- Compilação com `adam` e função perda `binary_crossentropy`
- Callbacks:
  - `ModelCheckpoint` para salvar o melhor modelo durante o treino
  - `EarlyStopping` para interromper o treino quando a validação não melhora
- Treinamento por até 15 épocas com validação.

---

### 4. Avaliação do modelo

- Plotagem dos gráficos de acurácia e perda para treino e validação por época.
- Predição no conjunto de teste.
- Cálculo e plot da matriz de confusão.
- Impressão do relatório de classificação com métricas detalhadas (precisão, recall, f1-score).

---

### 5. Aplicação web com Streamlit

- Interface simples para upload de imagens de raio-x.
- Imagem pré-processada (redimensionada e normalizada) para ser input da CNN.
- Apresentação do resultado com a classe prevista (`Normal` ou `Pneumonia`) e confiança da predição.

---

## Como usar

### No Google Colab

1. Configure a API do Kaggle para baixar o dataset [Chest X-Ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
2. Baixe e descompacte o dataset na pasta `/content/chest_xray/`.
3. Execute as células do notebook para visualizar as imagens, treinar a CNN e avaliar o modelo.
4. Salve o melhor modelo gerado (`melhor_modelo.h5`).

---

### Localmente (para rodar a aplicação Streamlit)

1. Instale as dependências:

```bash
pip install tensorflow matplotlib scikit-learn streamlit pillow
