### Features

- O sistema deverá detetar caras sempre que alguém chegar perto;

- Para além de detetar as caras o sistema deverá ser capaz de reconhecer as várias pessoas da turma (ou do grupo). Para isso pode funcionar com uma base de dados pré-gravada. Deve também ser possível iniciar o sistema sem ter ainda informação sobre nenhuma pessoa;

- Deve ser possível visualizar a base de dados das pessoas em tempo real;

- O sistema deverá identificar as pessoas que reconhece, e perguntar sobre as pessoas desconhecidas;

- O sistema deve cumprimentar as pessoas que já conhece, dizendo "Hello ". Poderá utilizar uma ferramenta de \emph{text to speech}, por exemplo https://pypi.org/project/pyttsx3/ ;

- O sistema deverá fazer o seguimento das pessoas na sala e manter a identificação em cima das pessoas que reconheceu anteriormente, ainda que atualmente não seja possível reconhecê-las.



# Entidade

![](https://github.com/jotadateta/SAVI_2022_23/blob/main/ua.png)

Universidade de aveiro, Departamento de engenharia mecanica 

# How to
## install
1. Instalar Face recognition [aqui](https://www.geeksforgeeks.org/how-to-install-face-recognition-in-python-on-linux/).

## Metodo
1. Todas as imagens "pre-registadas" encontram-se na pasta /faces.
2. Com o face_recognition foi se ler todas estas imagens e respetivos nomes de ficheiro.
3. Criou-se um programa para perceber a web-cam
4. Realizou-se o processamneto de imagem
5. Utilizou-se o face_recognition na imagem processada pela webcam
6. Processou-se a informação dando match com o "dataset" em caso de existencia
7. De seguida implementou-se um tracking para cada pessoa detectada


##### Todo o codigo encontra-se comentado de forma a facilitar a sua compreensão
