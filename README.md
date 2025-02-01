# **VideoHighlightDetection**

Nesse repositório se encontra tudo que é útil para realizar os experimentos

## **CSTA**

Em CSTA se encontra o modelo e o código para realização dos experimentos

Para treinar o modelo do 0, utilize o seguinte comando dentro da pasta CSTA
python3 train.py --datasets <TVSum or HiSum350>

Para verificar os resultados, utilize o seguinte comando dentro da pasta CSTA
python3 inference_hl.py --datasets <TVSum or HiSum350>

Para realizar os experimentos é necessário ter os arquivos h5 dos datasets na pasta data, dentro do diretório CSTA

https://drive.google.com/drive/folders/1bNib6LJVBwkSRhxB-w0qjys2cAaVMNZl?usp=drive_link

Para verificar os resultados é necessário ter os pesos do modelo pré-treinado na pasta weights, dentro do diretório CSTA
Você pode adquirir esses pessos no arquivo weights.tar, inclusos dentro desse diretório

https://drive.google.com/drive/folders/1bNib6LJVBwkSRhxB-w0qjys2cAaVMNZl?usp=drive_link


### **Os experimentos são realizados em 3 cenários**
- **hl**: padrão, não utiliza o incomodo psicoacústico em nenhuma etapa.
- **hlpa**: o incomodo psicoacústico é agregado ao valor estimado pelo preditor por meio da média aritimética.
- **hlunpa**: o incomodo psicoacústico é utilizado como ground truth no treinamento do modelo.


## **dataUtils**

Tudo que é necessário para criar o dataset

### Para aquisição do incomodo psicoacústico verificar minha implementação do modelo de zwicker

https://github.com/joesty/PYPAAnalyzer


## **Dependencias**
recomendo utilizar python3.10
para instalar basta usar pip install -r requirements.txt

## Autor:

Gustavo "Joesty" Ribeiro
