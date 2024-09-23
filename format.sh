
# check if the pre-commit command line tool is installed
if ! command -v pre-commit &> /dev/null
then
    echo "pre-commit is not installed. Please install it by running 'pip install -r requirements-lint.txt'."
    exit 1
fi

# if the .git/hooks/pre-commit file does not exist, install the pre-commit hooks
if [ ! -f .git/hooks/pre-commit ]; then
    echo "Installing pre-commit hooks..."
    pre-commit install --install-hooks
fi

pre-commit run --all-files

isort .