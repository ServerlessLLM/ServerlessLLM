# ServerlessLLM documents

Please find our documents in [ServerlessLLM](https://serverlessllm.github.io/docs/intro).

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

Images are stored in the `images` directory alongside the documentation files. This ensures that each documentation version has its own images, preventing version conflicts.

**Usage:**
- Store images in: `docs/images/` (for Latest) or `versioned_docs/version-X.X.X/images/` (for versioned docs)
- Reference images using relative paths: `./images/filename.jpg`

**Examples:**
```markdown
<!-- Markdown syntax (no width control) -->
![Alt text](./images/a.jpg)

<!-- HTML syntax with sizing (use require for Docusaurus) -->
<img src={require('./images/a.jpg').default} alt="Description" width="256px"/>
```

When you create a new documentation version using `npm run docusaurus docs:version X.X.X`, the images directory is automatically copied with the versioned docs.
