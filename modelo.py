import pandas as pd
import urllib.request
import statsmodels.formula.api as smf

#URL do arquivo
url = 'https://raw.githubusercontent.com/PHCJ/DataBaseModeloTesteOLS/main/sales_data.csv'
#Nome e o caminho do arquivo de destino para salvar o arquivo baixado
filename = 'sales_data.csv'
#Função em Python que é usada para baixar arquivos da internet
urllib.request.urlretrieve(url, filename)
#Carregar os dados de um arquivo CSV para um dataframe do pandas:
df = pd.read_csv('sales_data.csv')

# Relação entre as variáveis independentes (frequência de compra e histórico do cliente) e a variável dependente (vendas)
formula = 'vendas ~ frequencia_de_compra + historico_do_cliente'

# Ajusta o modelo 
model = smf.ols(formula=formula, data=df).fit()

# Novos dados para previsão
new_data = {'frequencia_de_compra': [10, 15, 6],
            'historico_do_cliente': [1000, 1500, 300]}
#Previsões sobre os novos dados fornecidos no dicionário acima
previsao = model.predict(pd.DataFrame(new_data))

print(previsao)