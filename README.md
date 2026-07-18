# MIST_Download
Este repositório é para que aqueles que desejarem baixar nossa ferramenta MIST.

## Pré-requisitos:

Antes de começar, você precisará ter instalado em sua máquina:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/) (incluído no Docker Desktop)
* Git

## Passos para rodar a ferramenta:

- Clone este repositório no diretório que deseje.

### Passo 1: Obter a API Key da Prior Labs

1. Acesse https://ux.priorlabs.ai
2. Faça login ou registre-se
3. Aceite a licença na aba **Licenses**
4. Copie sua API Key em https://ux.priorlabs.ai/api/keys

### Passo 2: Configurar o Environment

Crie um arquivo chamado exatamente `.env` na raiz do projeto e coloque:

```bash
PRIORLABS_API_KEY=sua_api_key_tabpfn_aqui
```

Substitua o valor pela API Key obtida no site acima.

> Observação: o `.env` e o `Dockerfile` já estão configurados para usar essa API Key durante a execução do container.

### Passo 3: Executar o Docker

Com a API Key configurada e o Docker inicializado, execute:

```bash
docker compose up -d --build
```

Após o build e execução:

- Acesse `localhost:3000`
- Para parar a aplicação:

```bash
docker compose down
```