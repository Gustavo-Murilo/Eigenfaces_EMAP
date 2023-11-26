import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Constante com a dimensão da imagem
DIMENSAO_IMAGEM = (100,100)
FACE_MEDIA = np.zeros(DIMENSAO_IMAGEM)
NOME_AMBIENTE = ""
contador_imagem = 0

def tratar_imagem(caminho: str) -> np.ndarray :
    # Carrega a imagem em uma variável
    imagem = Image.open(caminho)

    # Altera as dimensões para o padrão de PIL (largura, altura)
    dim_imagem = tuple(reversed(DIMENSAO_IMAGEM))
    
    # Redimensiona a imagem
    imagem = imagem.resize((dim_imagem))
    # Converte a imagem para escala de cinza
    imagem = imagem.convert('L')
    
    # Converte a imagem para um vetor
    vetor_imagem = np.array(imagem)
    # Torna o vetor unidmentsinal e o normaliza
    vetor_imagem = vetor_imagem.flatten() / 255.0
    
    return vetor_imagem


def plotar_imagem(vetor_imagem: np.ndarray,) -> None :
    # Transforma o vetor unidmensional em uma matriz (m x n)
    imagem = np.reshape(vetor_imagem, (DIMENSAO_IMAGEM))

    # Configurações para a plotagem da imagem
    plot = plt.imshow(imagem, cmap='gray')
    plt.axis('off')

    # Salva a imagem 
    global contador_imagem
    plt.savefig(f'{NOME_AMBIENTE}_{contador_imagem}.png')
    contador_imagem += 1
    
    plt.show()

    return None


def varrer_banco_imagens(caminho: str) -> (int, np.ndarray) :
    # Gera uma lista com os vetores das faces
    banco_imagens = []
    # Instancia um contador
    quantidade_imagens = 0
    
    for arquivo in glob.glob(caminho):
        banco_imagens.append(tratar_imagem(arquivo))
        quantidade_imagens += 1
        
    # Converte a lista para uma matriz (q x m*n), sendo 'q' a quantidade de imagens
    matriz_imagens = np.array(banco_imagens)
    # Transpoe a matriz, que passa a ser (m*n x q)
    matriz_imagens = matriz_imagens.T
    
    return (quantidade_imagens, matriz_imagens)


def plotar_grade(matriz: np.ndarray, descricao: str, qtd_linhas: int, qtd_colunas: int, tamanho: tuple, titulo: str) -> None :
    # Definição de uma grade para a plotagem de imagens
    fig, axes = plt.subplots(qtd_linhas, qtd_colunas, figsize=tamanho)

    if titulo != '' :
        fig.suptitle(titulo, fontsize=16)
    
    for indice, ax in enumerate(axes.flat):
        # Definição da imagem a ser plotada
        matriz_imagem = matriz[:, indice].reshape(DIMENSAO_IMAGEM)

        # Ajustes para a plotagem
        ax.imshow(matriz_imagem, cmap=plt.cm.gray)
        ax.axis('off')
        if isinstance(descricao, str) :
            ax.set_title(f'{descricao} {indice + 1}')
        else:
            ax.set_title(f'k = {descricao[indice]}')

    # Salva a imagem 
    global contador_imagem
    plt.savefig(f'{NOME_AMBIENTE}_{contador_imagem}.png')
    contador_imagem += 1
    
    plt.show()
    
    return None


def plotar_grade_alternada(matrizes: np.ndarray, descricao: str, indices_visible: bool, qtd_linhas: int, qtd_colunas: int, tamanho: tuple, titulo: str) -> None :
    # Definição de uma grade para a plotagem de imagens
    fig, axes = plt.subplots(qtd_linhas, qtd_colunas, figsize=tamanho)

    # Insere título na grade se for o caso
    if titulo != '' :
        fig.suptitle(titulo, fontsize=16)

    # Adequa o programa à quantidade de matrizes
    qtd_matrizes = 1
    if isinstance(matrizes, tuple) :
        qtd_matrizes = len(matrizes)
        
    for i, ax in enumerate(axes.flat):
        indice = i // qtd_matrizes
        
        # Definição da imagem a ser plotada
        escolhida = i % qtd_matrizes
        matriz = matrizes[escolhida] 
        
        # Transforma o vetor unidimensional em uma matriz com as dimensões da imagem
        matriz_imagem = matriz[:,indice].reshape(DIMENSAO_IMAGEM)
    
        # Ajustes para a plotagem
        ax.imshow(matriz_imagem, cmap=plt.cm.gray)
        ax.axis('off')

        legenda = f'{descricao[escolhida]}'
        if indices_visible : legenda += f' {indice}'
        ax.set_title(legenda)

    # Salva a imagem 
    global contador_imagem
    plt.savefig(f'{NOME_AMBIENTE}_{contador_imagem}.png')
    contador_imagem += 1
    
    plt.show()
   
    return None

def projetar_matriz(eigenfaces: np.ndarray, coeficientes: np.ndarray, dimensao: int, imagem_especifica: int) -> np.ndarray :
    # Reconstrução da imagem usando as eigenfaces
    matriz = np.dot(eigenfaces[:, :dimensao], coeficientes[:dimensao, :]) + FACE_MEDIA

    if isinstance(imagem_especifica, int) :
        matriz = matriz[:, imagem_especifica]
    
    return matriz

def listar_projecoes(eigenfaces: np.ndarray, coeficientes: np.ndarray, intervalo: list, imagem_especifica: int) -> np.ndarray :

    lista = []
    
    for dimensao in intervalo :
        # Reconstrução da imagem usando as eigenfaces
        matriz_reconstruida = projetar_matriz(eigenfaces, coeficientes, dimensao, imagem_especifica)

        # Salvar imagem em uma lista
        lista.append(matriz_reconstruida)

    # Converte a lista para uma matriz
    lista_proj = np.array(lista).T
    
    return lista_proj

def exibir_memoria_ocupada(lista_var: list) -> None :
    
    lista_nomes = ['A', 'M', 'C', 'U_A_k100', 'coef_A']

    lista_magnitude = ['mb', 'gb']
    
    for i in range(len(lista_nomes)):
        ocupacao = ''
    
        for j in range(2):
            tamanho = f'{(lista_var[i].size * lista_var[i].itemsize) / (1024.0 ** (j+2)):_.1f}'.replace(".",",").replace("_",".")
            ocupacao = ocupacao + f'{tamanho} {lista_magnitude[j]} = '

        print(f'{lista_nomes[i]} ocupa {ocupacao[:-2]}')
    
    return None

def plotar_relevancia_eigenfaces(autovalores: np.ndarray, titulo: str) -> None :
    soma_total = autovalores.sum()

    # Vetor com a relevância (percentual) de cada autovalores
    relevancia = autovalores / soma_total
   
    # Plotar o percentual de valores em um gráfico de barras
    plt.plot(relevancia, marker='o', linestyle='-')
    plt.xlabel('Autovetor')
    plt.ylabel('Percentual')
    plt.title(f'Relevância das eigenfaces {titulo}')

    # Salva a imagem 
    global contador_imagem
    plt.savefig(f'{NOME_AMBIENTE}_{contador_imagem}.png')
    contador_imagem += 1

    plt.show()
    
    return None