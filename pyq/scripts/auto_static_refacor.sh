python -m black -l 120 ./pyq
python -m isort --line-length=120 ./pyq
python -m flake8 --max-line-length=120 ./pyq
