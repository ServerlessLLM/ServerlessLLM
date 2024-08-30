name: Bug Report
description: File an issue about a bug in ServerlessLLM.
title: "[BUG] "
labels: [bug]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Please provide as much detail as possible to help us address the issue efficiently. If you're unsure if this is a bug, consider asking in [Discussions](https://github.com/ServerlessLLM/ServerlessLLM/discussions) first.

  - type: checkboxes
    id: prerequisites
    attributes:
      label: Prerequisites
      options:
        - label: I have read the [ServerlessLLM documentation](https://serverlessllm.github.io/).
          required: true
        - label: I have searched the [Issue Tracker](https://github.com/ServerlessLLM/ServerlessLLM/issues) and [Discussions](https://github.com/ServerlessLLM/ServerlessLLM/discussions) to ensure this hasn't been reported before.
          required: true

  - type: textarea
    id: system-info
    attributes:
      label: System Information
      description: Please provide details about your environment (OS, Python version, GPU, etc.).
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Problem Description
      description: Provide a clear description of the bug.
    validations:
      required: true

  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Please provide code snippets and steps to reproduce the issue.
      value: |
        Code snippets:
        ```python

        ```

        Steps to reproduce:
        1.
        2.
        3.
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What did you expect to happen?

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Add any other relevant information, screenshots, or suggested fixes.