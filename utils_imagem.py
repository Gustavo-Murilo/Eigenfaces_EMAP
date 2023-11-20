import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Constante com a dimensão da imagem
DIMENSAO_IMAGEM = (100,100)

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

    plt.show()
    
    return None


# NÂO USADA AINDA
# def reconstruir_imagem(Matriz: np.ndarray) -> np.ndarray :
#     # Coeficientes de projeção para as eigenfaces da Matriz
#     coeficientes_C = np.dot(U_C.T, M)
#
#     # Reconstrução da imagem usando as eigenfaces de U_A
#     imagem_reconstruida_C = np.dot(U_C, coeficientes_C) + face_media
#
#     # Remodelar a imagem reconstruída para as dimensões originais
#     imagem_reconstruida_C = imagem_reconstruida_C[:,1].reshape(DIMENSAO_IMAGEM)
#     return np.array([1])