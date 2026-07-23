# MIST_Download
Este repositório é para que aqueles que desejarem baixar nossa ferramenta MIST.

## Pré-requisitos:

Antes de começar, você precisará ter instalado em sua máquina:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/) (incluído no Docker Desktop)
* Git

## Passos para rodar a ferramenta:
### Clonar repositório

- Clone este repositório no diretório que deseje.

### Executar o Docker

Execute:

```bash
docker compose up -d --build
```

Após o build e execução:

- Acesse `localhost:3000` para utilizar a ferramenta
- Para parar a aplicação:

```bash
docker compose down
```
