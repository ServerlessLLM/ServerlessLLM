# Contributing to ServerlessLLM

Thank you for considering contributing to ServerlessLLM!
We welcome contributions of all kinds from the community.
Whether you're introducing new features, enhancing the infrastructure, fixing bugs, or writing documentation, we appreciate your enthusiasm and value your efforts.

To help make your contributions as smooth as possible, we've put together this guide with helpful tips and best practices for contributing to the project.

## Table of Contents

- [Join the Community](#join-the-community)
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
   - [Roadmap and Tasks](#roadmap-and-tasks)
   - [Development Environment](#development-environment)
   - [Commit Message Guidelines](#commit-message-guidelines)

## Join the Community

- Discord: [ServerlessLLM Discord](https://discord.gg/AEF8Gduvm8)

## Code of Conduct

We expect all contributors to follow our [Code of Conduct](https://github.com/ServerlessLLM/ServerlessLLM/blob/main/CODE_OF_CONDUCT.md). Be respectful, inclusive, and open to feedback.

## How to Contribute

- Check the [public development board](https://github.com/orgs/ServerlessLLM/projects/2) for tasks (i.e., items with the status `Ready 🟢`) or [issue tracker](https://github.com/ServerlessLLM/ServerlessLLM/issues) for open issues. You can also create a new one to discuss your idea.
- Follow the [Fork-and-Pull-Request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow when opening your pull requests.
- Ensure your code follows our style guidelines and passes all tests (see [Development Environment](#development-environment)).
- Submit a pull request with a clear description of your changes.
  - The pull request title should follow the [Commit Message Guidelines](#commit-message-guidelines).
  - The description should follow the [Pull Request Template](https://github.com/ServerlessLLM/ServerlessLLM/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
  - Make sure to mention any related issues.

Before your pull request can be merged, it must pass the formatting, linting, and testing checks (see [Development Environment](#development-environment)).

If you fix a bug:
- Add a relevant unit test when possible. These can be found in the `test` directory.
If you make an improvement:
- Update any affected example console scripts in the `examples` directory and documentation in the `docs` directory.
- Update unit tests when relevant.
If you add a feature:
- Include unit tests in the `test` directory.
- Add a demo script in the `examples` directory.

### Roadmap and Tasks

You can find available tasks (i.e., `status=Ready 🟢`) and contribute to planned features by checking our [public development board](https://github.com/orgs/ServerlessLLM/projects/2).
For beginners, we recommend starting with issues labeled `good first issue` or `help wanted`.
Feel free to discuss any ideas before getting started!

### Development Environment

Ensure your development environment is set up with the following tools:

- Format your code with pre-commit hooks:
```bash
pip install -r requirements-lint.txt

# add lint hooks to git commit
pre-commit install --install-hooks
```

This will automatically format your code before committing. However, you can also run the following commands manually:
```bash
# format code
pre-commit run -a
```

- (Recommended) Sign off your commits:
```bash
git commit -s -m "feat: add new feature"
```

### Commit Message Guidelines

We follow the commit format rule based on the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format). This format improves readability and helps generate changelogs automatically.

#### Commit Message Structure

Each commit message should consist of a **header** and a **body**:

```
<type>: <summary>
<BLANK LINE>
<body>(optional)
<BLANK LINE>
```
- **Type**: Choose from `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `test`, `chore`.
- **Summary**: A brief description of the change.
- **Body**: Mandatory for all commits except those of type "docs". Must be at least 20 characters long.


Examples:

```
feat: add logging in sllm worker
```

```
docs: add new example for serving vision model

Vision mode: xxx
Implemented xxx in `xxx.py`
```

For more details, read the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format).
