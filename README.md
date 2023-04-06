# Segmentação de Clientes

****

## 🔍 Sobre o projeto

Nós trabalhamos em uma empresa de telecomunicações e um dos problemas da empresa é a alta taxa de churn. Para resolver isso, solicitamos uma base de dados com informações de diversos clientes anteriores para construir uma solução utilizando Machine Learning e prever se um futuro cliente irá dar churn ou não. Ao final, vamos ver se o nosso modelo é capaz de gerar lucro para a companhia ou não.

Agora, vamos fazer a apresentação do dataset que iremos utilizar:

- <strong>customerID</strong>: Código de identificação único do cliente
- <strong>gender</strong>: Gênero do cliente (Male ou Female)
- <strong>SeniorCitizen</strong>: Idoso ou não (1 ou 0)
- <strong>Partner</strong>: Possui um parceiro(a) (Yes ou No)
- <strong>tenure</strong>: Número de meses que o cliente ficou com a companhia
- <strong>PhoneService</strong>: Possui serviço de telefone (Yes ou No)
- <strong>MultipleLines</strong>: Possui múltiplas linhas de telefone (No phone service, No ou Yes)
- <strong>InternetService</strong>: Serviço de internet (DSL, Fiber optic, No)
- <strong>OnlineSecurity</strong>: Serviço de segurança online (No, Yes ou No internet service)
- <strong>OnlineBackup</strong>: Serviço de backup online (No, Yes ou No internet service)
- <strong>DeviceProtection</strong>: Proteção do dispositivo (No, Yes ou No internet service)
- <strong>TechSupport</strong>: Assistência técnica (No, Yes ou No internet service)
- <strong>StreamingTV</strong>: Possui serviço de streaming (No, Yes ou No internet service)
- <strong>StreamingMovies</strong>: Possui serviço de filmes em streaming (No, Yes ou No internet service)
- <strong>Contract</strong>: Tipo de contrato (Month-to-month, One year ou Two year)
- <strong>PaperlessBilling</strong>: Cobrança de conta sem papel (Yes ou No)
- <strong>PaymentMethod</strong>: Método de pagamento (Electronic check, Mailed check, Bank transfer ou Credit card)
- <strong>MonthlyCharges</strong>: Recarregamento mensal
- <strong>TotalCharges</strong>: Recarregamento total
- <strong>churn</strong>: Cliente deixou a companhia (churn) (Yes ou No)

## 🗃️ Tópicos do projeto

O projeto é dividido nos seguintes tópicos:
<ol>
  <li> Importando bibliotecas e dataframe
  <li> Explorando o dataset
  <li> Cluster com o algoritmo K-Means
  <li> Análise Exploratória de Dados (Univariada e Bivariada)
  <li> Aplicando K-Means com PCA
  <li> Modelagem
  <li> Análise de negócio
</ol>
