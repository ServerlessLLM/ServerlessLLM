
# check if the pre-commit command line tool is installed
if ! command -v pre-commit &> /dev/null
then
    echo "pre-commit is not installed. Please install it by running 'pip install -r requirements-lint.txt'."
    exit 1
fi
