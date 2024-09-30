# Contributing to ServerlessLLM

# Contributors

ServerlessLLM is currently maintained by the following contributors:

 - [Yao Fu](https://github.com/future-xy)
 - [Leyang Xue](https://github.com/drunkcoding)
 - [Yeqi Huang](https://github.com/Chivier)
 - [Andrei-Octavian Brabete](https://github.com/andrei3131)
 - [Matej Sandor](https://github.com/MatejSandor)
 - [Ruiqi Lai](https://github.com/lrq619)
 - [Siyang Shao](https://github.com/SiyangShao)
 - [Xinyuan Tong](https://github.com/JustinTong0323)
 - [Luo Mai](https://github.com/luomai)

## Code Format

```bash
pip install -r requirements-lint.txt

# add lint hooks to git commit
pre-commit install --install-hooks
```

## Commit Message Guidelines

We follow the commit format rule based on the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format). This format improves readability and helps generate changelogs automatically.

### Commit Message Structure

Each commit message should consist of a **header**, **body**, and **footer**:

```
<type>(<scope>): <short summary>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

- **Header**: Mandatory. Contains the commit type, optional scope, and a brief summary.
- **Body**: Mandatory for all commits except those of type "docs". Must be at least 20 characters long.
- **Footer**: Optional. Used for breaking changes, deprecations, or references to issues/PRs.

### Commit Message Header

The header must follow this format:

```
<type>(<scope>): <short summary>
  │       │             │
  │       │             └─⫸ Summary in present tense. Not capitalized. No period at the end.
  │       │
  │       └─⫸ Commit Scope: serve|store|cli
  │
  └─⫸ Commit Type: build|ci|docs|feat|fix|perf|refactor|test|chore
```

- **type**: Specifies the nature of the commit. Must be one of the following:
  - `build`: Changes that affect the build system or external dependencies
  - `ci`: Changes to our CI configuration files and scripts
  - `docs`: Documentation only changes
  - `feat`: A new feature
  - `fix`: A bug fix
  - `perf`: A code change that improves performance
  - `refactor`: A code change that neither fixes a bug nor adds a feature
  - `test`: Adding missing tests or correcting existing tests
  - `chore`: Minor editing jobs (e.g., updating README)
- **scope**: Optional. Specifies the area of the codebase affected. For example:
  - serve
  - store
  - cli
- **summary**: A brief description of the change. Must:
  - Be in the imperative, present tense: "change" not "changed" nor "changes"
  - Not capitalize the first letter
  - Not end with a period

### Commit Message Body

The body should explain the motivation for the change and contrast it with the previous behavior. It must:

- Use the imperative, present tense: "fix" not "fixed" nor "fixes".
- Be at least 20 characters long.

### Commit Message Footer

The footer can contain additional information such as breaking changes, deprecations, or references to issues/PRs. Use the following formats:

**Breaking Changes**:

```
<TEXT>
BREAKING CHANGE: <breaking change summary>
<BLANK LINE>
<breaking change description + migration instructions>
<BLANK LINE>
<BLANK LINE>
Fixes #<issue number>
```

**Deprecations**:

```
<TEXT>
DEPRECATED: <what is deprecated>
<BLANK LINE>
<deprecation description + recommended update path>
<BLANK LINE>
<BLANK LINE>
Closes #<pr number>
```

### Revert Commits

If reverting a previous commit, the message should start with `revert:` followed by the header of the reverted commit. The body should contain:

- Information about the SHA of the commit being reverted in the format: `This reverts commit <SHA>`
- A clear description of the reason for reverting the commit.

## Pull Requests

When contributing to the repository, you should work in a separate branch and create a GitHub pull request for your branch. For all pull requests to `ServerlessLLM`, we require that you do the following:

### Sync Your Repo

When working on a fork of the `ServerlessLLM` repository, keeping your fork in sync with the main repository keeps your workspace up-to-date and reduces the risk of merge conflicts.

1. If you have not done so already, create a new remote for the upstream `ServerlessLLM` repo:

   ```bash
   git remote add upstream https://github.com/your-organization/ServerlessLLM.git
   ```

2. You can always check your existing remotes with:

   ```bash
   git remote -v
   ```

3. Fetch branches and commits from the upstream (ServerlessLLM) repo:

   ```bash
   git fetch upstream
   ```

4. Switch to your local default branch (named `main` by default):

   ```bash
   git checkout main
   ```

5. Merge the upstream changes:

   ```bash
   git merge upstream/main
   ```

For more information, check out the official GitHub docs on [syncing forks](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork).

### Commit Sign-off

Maintaining a clear and traceable history of contributions is essential for the integrity and accountability of our project. To achieve this, we require that all contributors sign off on their Git commits. This process ensures that you, as a contributor, acknowledge and agree to the terms of our project's licensing and contribution guidelines.

#### How to Add a Sign-off

To add a sign-off to your commit message, you can use the `-s` or `--signoff` flag with the `git commit` command:

```bash
git commit -s -m "Your commit message"
```

Alternatively, you can manually add the sign-off line to your commit message, like this:

```
Your commit message

Signed-off-by: Your Name <your.email@example.com>
```

#### Consequences of Not Signing Off

Commits that do not include a valid sign-off will not be accepted into the main branch of the repository. Failure to comply with this requirement may result in the rejection of your contributions.

### Squashing Commits and Merging

We maintain a clean and meaningful commit history on the main branch by ensuring each merged pull request represents a single, cohesive change. To achieve this, we use GitHub's "Squash and merge" feature.

#### Why Squash and Merge?

Squashing commits before merging offers several advantages:

1. **Clean History**: The main branch maintains a clear, linear history where each commit represents a complete feature or fix.
2. **Simplified Understanding**: It's easier for contributors to grasp the project's evolution by reading concise, feature-level commit messages.
3. **Easier Reverting**: If needed, reverting an entire feature becomes straightforward as it's contained in a single commit.
4. **Preserved Details**: The full commit history of the feature development is retained in the pull request for future reference.
5. **Reduced Noise**: Intermediate commits, including "work in progress" or "fix typo" commits, are consolidated into a single, meaningful commit.

#### Workflow Example

Let's walk through an example of adding a new checkpoint format:

1. Create and switch to a new feature branch:

   ```bash
   git checkout -b feature/add-new-checkpoint-format
   ```

2. Make changes and commit them:

   ```bash
   # Implement new checkpoint format
   git add .
   git commit -m "Add basic structure for new checkpoint format"

   # Add serialization method
   git add .
   git commit -m "Implement serialization for new format"

   # Add deserialization method
   git add .
   git commit -m "Implement deserialization for new format"

   # Fix a bug in serialization
   git add .
   git commit -m "Fix endianness issue in serialization"
   ```

3. Push your branch and create a pull request on GitHub.

4. After the review process and any necessary changes, the maintainer will use the "Squash and merge" option.

5. The resulting commit on the main branch will look like this:

   ```
   Add new checkpoint format (#78)

   This pull request implements a new checkpoint format, including:
   - Basic structure for the new format
   - Serialization method with correct endianness
   - Deserialization method

   The new format improves storage efficiency and load times.

   Squashed commit of the following:
   - Add basic structure for new checkpoint format
   - Implement serialization for new format
   - Implement deserialization for new format
   - Fix endianness issue in serialization
   ```

#### How to Squash and Merge

When a pull request is ready to be merged:

1. Go to the pull request page on GitHub.
2. Click the "Merge pull request" dropdown and select "Squash and merge".
3. Edit the commit message to provide a clear, concise summary of the changes.
4. Click "Confirm squash and merge".

#### After Merging

After your pull request is merged:

1. Delete your local feature branch:

   ```bash
   git checkout main
   git branch -d feature/add-new-checkpoint-format
   ```

2. Update your local main branch:

   ```bash
   git pull origin main
   ```

3. Delete the remote feature branch:

   ```bash
   git push origin --delete feature/add-new-checkpoint-format
   ```

By following this workflow, we maintain a clean and organized main branch, making it easier for all contributors to understand the project's history and collaborate effectively. The detailed development process remains available in the pull request history, providing the best of both worlds: a clean main branch and preserved development details.

## Release
### Release with Github Workflow
1. Bump the version number in `ServerlessLLM/setup.py` and `ServerlessLLM/serverless_llm/setup.py`
```
setup(
    name=...,
    version="<version-number>",
    ext_modules=...
    ...
)
```
2. Tag the current commit and push:
```
git tag v<x.x.x>
git push origin v<x.x.x>
``` 

### Release Manually
1. Build the package in an NVIDIA container.

   ```bash
   docker build -t sllm_store_builder -f Dockerfile.builder .
   docker run -it --rm -v $(pwd)/dist:/app/dist sllm_store_builder /bin/bash
   # TODO: use workflow to build the package.
   export PYTHON_VERSION=310
   export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
   conda activate py${PYTHON_VERSION} && python setup.py sdist bdist_wheel
   ```

2. Upload the package to TestPyPI.

   ```bash
   pip install twine
   # NOTE: rename "linux" to "manylinux1" in the generated wheel file.
   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

3. Install the package from TestPyPI.

   ```bash
   # Create a virtual environment
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ serverless-llm-store==0.0.1dev4
   ```
