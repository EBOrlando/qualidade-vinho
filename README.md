# qualidade-vinho
Case realizado para cognitivo.ai - Previsão da qualidade do vinho

Para definição do processo de modelagem primeiro foi necessário o entendimento do problema posto, uma vez feito isso podemos seguir com:

1. Analise dos dados em busca de outliers ou outros problemas similares
2. Breve analise de correlação entre as variáveis explicativas e a qualidade do vinho
3. Processo de modelagem
4. Validação do resultado

Neste caso em específico, decidi por dividir os vinhos em 3 qualidades:
1. Ruim: 0 a 4
2. Normal: 5 a 7
3. Excelente: 8 a 10

Assim posso trabalhar com modelos de classificação, neste caso: Floresta Aleatória, SVM e Regressão Logística.
Utilizaremos como "função de custo" a acurácia, selecionando o modelo com a maior acurácia como decisor.
Gostaria de utilizar uma validação out of time para confirmar minha decisão, mas como isso não é possível utilizarei a técnica k fold cross-validation comparando as acurácias médias, essa técnica se mostra boa pois é capaz de detectar caso haja overfitting.

O modelo selecionado se mostra suficientemente bom quando analisamos o panorama geral.
Entretanto devido ao baixo volume de observações classificadas como "ruim, excelente" o modelo tem sua taxa de acertividade reduzida para essas classes.

