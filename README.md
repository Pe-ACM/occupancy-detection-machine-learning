# Detecção de Ocupação de Escritório

Projeto de classificação supervisionada usando o dataset **Occupancy Detection**, da UCI Machine Learning Repository.

## Estrutura
- `datatraining.txt`: conjunto oficial de treinamento.
- `datatest.txt` e `datatest2.txt`: conjuntos oficiais de teste.
- `analise_occupancy.py`: análise exploratória, treinamento e avaliação.
- `resultados_occupancy/`: métricas, relatórios por classe, matrizes e gráficos.

## Execução
```bash
pip install -r requirements.txt
python analise_occupancy.py
```

## Modelos
- Árvore de Decisão
- Random Forest
- K-Nearest Neighbors (KNN), com padronização das variáveis

## Metodologia
Os modelos são treinados em `datatraining.txt` e avaliados na união de `datatest.txt` e `datatest2.txt`, preservando a divisão temporal oficial do repositório. O identificador e a data não são usados como preditores. A classe é `Occupancy` (0 = não ocupado; 1 = ocupado).

## Fonte
CANDANEDO, Luis. Occupancy Detection. UCI Machine Learning Repository, 2016. DOI: 10.24432/C5X01N.
