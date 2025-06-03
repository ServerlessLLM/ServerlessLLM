# ServerlessLLM documents

Please find our documents in [ServerlessLLM](https://serverlessllm.github.io/docs/stable/getting_started).

## How to build ServerlessLLM Docs

This website is built using Docusaurus, a modern static website generator.

### Installation

To install the necessary dependencies, use the following command:

```bash
npm install
```

### Local Development

To start a local development server and open up a browser window, use the following command:

```bash
npm run start
```

Most changes are reflected live without having to restart the server.

### Build

To generate static content into the build directory, use the following command:

```bash
npm run build
```

This command generates static content into the `build` directory, which can be served using any static content hosting service.

### About the image path

Images are stored in `images` path. For example, we have an image called `a.jpg` in `images`. When we use this image in any position in the documents, we just use `/img/a.jpg`. (The document sync bot can copy `images` path into `img` folder in `serverlessllm.github.io` repo)
