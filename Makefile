install:
	@pip install --upgrade pip && pip install -r requirements.txt

lint:
	pylint --disable=R,C,logging-fstring-interpolation,f-string-without-interpolation --fail-under=7.0 src/*.py *.py *.ipynb 

test:
	# --cov=my_function test_*.py for details on a function
	# test_*.py → python tests files are prefixed as is
	#  --disable-warnings, if needed
	# -vvv for verbose
	# -s for disabling capturing
	#     (allows print statements to be shown in the console even if the test passes)
	@python -m pytest -vvv -s

format:
	# --force-exclude '<FILE_OR_FOLDER>' if needed (env, imported, models...)
	@black src/*.py *.py *.ipynb

all: install lint test format