sh scripts/clean.sh
python setup.py sdist bdist_wheel
twine upload dist/*
