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

## Check Code Format

```bash
pip install ruff
pip install isort

# Format Python import packages
isort .

# Check code format
ruff check

# Run code format
ruff format
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

When contributing to the repository, you should work in a separate branch and create a GitHub pull request for your branch. For all pull requests to `serverlessLLM`, we require that you do the following:

### Sync Your Repo

When working on a fork of the `serverlessLLM` repository, keeping your fork in sync with the main repository keeps your workspace up-to-date and reduces the risk of merge conflicts.

1. If you have not done so already, create a new remote for the upstream `serverlessLLM` repo:

   ```bash
   git remote add upstream https://github.com/your-organization/serverlessLLM.git
   ```

2. You can always check your existing remotes with:

   ```bash
   git remote -v
   ```

3. Fetch branches and commits from the upstream (serverlessLLM) repo:

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

### Squashing Commits

We prefer for every commit in the repo to encapsulate a single concrete and atomic change/addition. This means our commit history shows a clear and structured progression and remains concise and readable. In your own branch, you can clean up your commit history by squashing smaller commits together using `git rebase`:

1. Find the hash of the oldest commit which you want to squash up to. For example, if you made three commits in sequence (A, B, C, such that C is the latest commit) and you wanted to squash B and C then you would need to find the hash of A. You can find the hash of a commit on GitHub or by using the command:

   ```bash
   git log
   ```

2. Use the rebase command in interactive mode:

   ```bash
   git rebase -i [your hash]
   ```

3. For each commit which you would like to squash, replace "pick" with "s". Keep in mind that the "s" option keeps the commit but squashes it into the previous commit, i.e. the one above it. For example, consider the following:

   ```
   pick 4f3d934 commit A
   s c24c160 commit B
   s f20ac90 commit C
   pick 7667d38 commit D
   ```

   This would squash commits A, B, and C into a single commit, and then commit D would be left as a separate commit.

4. Update the commit messages as prompted.

5. Push your changes:

   ```bash
   git push --force
   ```

Apart from squashing commits, `git rebase -i` can also be used for rearranging the order of commits. If you are currently working on a commit and you already know that you will need to squash it with the previous commit at some point in the future, you can also use `git commit --amend` which automatically squashes with the last commit.

### Rebasing on the Main Branch

Rebasing is a powerful technique in Git that allows you to integrate changes from one branch into another. When we rebase our branch onto the main branch, we create a linear history and avoid merge commits. This is particularly valuable for maintaining a clean and structured commit history.

#### Why Rebase?

Consider a scenario where you have a branch (`feature`) that you started from the main branch, and both have received new commits since your branch was created:

```
      A---B---C feature
     /
D---E---F---G main
```

When you rebase your branch onto the main branch, Git rewrites the commit history. It moves the starting point of your branch to the tip of the main branch, making the history linear and eliminating the need for merge commits:

```
              A'--B'--C' feature
             /
D---E---F---G main
```

A messy merge without rebasing can result in a cluttered history:

```
        A---B---C feature
       /           \
D---E---F---G-------M main
```

#### How to Rebase on Main

To rebase on `main`, simply checkout your branch and use the following:

```bash
git checkout your-branch
git rebase main
```

This command sequence switches to your branch and reapplies your changes on top of the latest main branch. It's important to resolve any conflicts that may arise during the rebase process.

For more details and options, refer to the [git rebase documentation](https://git-scm.com/docs/git-rebase).

### Maintaining a Clean Main Branch & Avoiding Merge Commits

Maintaining a clean and linear history on your main branch is essential for several reasons:

- It simplifies the process of syncing the forked repository with the upstream repository.
- A clean main branch minimizes the likelihood of merge conflicts when you synchronize your fork with the upstream repository.
- A clean main branch enhances collaboration by making it easier for you and your collaborators to review and understand the project's history.

To avoid merge commits, follow the guidelines described above, particularly taking care to rebase on the main branch and keeping your forks in sync.

## Release

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
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ serverless-llm-store==0.0.1dev23
   ```
